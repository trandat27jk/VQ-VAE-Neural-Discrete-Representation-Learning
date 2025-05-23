import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import fsspec
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from VQ_VAE import vq_vae_loss


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    batch_size: int = 64
    data_loader_workers: int = os.cpu_count() // 2
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None
    beta: float = None
    max_grad_norm: float = None
    resume: str = None


@dataclass
class Snapshot:
    model_state: "OrderedDict[str,torch.tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class Trainer(nn.Module):
    def __init__(
        self,
        model,
        trainer_config: TrainerConfig,
        optimizer,
        train_dataset,
        test_dataset=None,
    ):
        super().__init__()
        self.trainer_config = trainer_config
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset)

        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = trainer_config.save_every
        self.epoch_run = 0
        if self.trainer_config.use_amp:
            self.scaler = torch.amp.GradScaler()
        if (
            self.trainer_config.snapshot_path is not None
            and self.trainer_config.resume == False
        ):
            self._load_snapshot()

        if self.trainer_config.snapshot_path is None:
            self.trainer_config.snapshot_path = "snapshot.pt"

    def _prepare_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=False,
            sampler=DistributedSampler(dataset),
            pin_memory=True,
            num_workers=self.trainer_config.data_loader_workers,
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec(self.trainer_config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("file not found, train from scratch")
            return
        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epoch_run = snapshot.finished_epoch
        print(f"Resume training at epoch {self.epoch_run}")

    def _run_batch(self, source, train=True):
        with torch.set_grad_enabled(train), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.trainer_config.use_amp),
        ):
            result = self.model(source)
            loss = vq_vae_loss(
                result["output"],
                source,
                result["ze"],
                result["e"],
                beta=self.trainer_config.beta,
            )
        if train:
            self.optimizer.zero_grad()
            if self.trainer_config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), self.trainer_config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), self.trainer_config.max_grad_norm
                )
                self.optimizer.step()
        return loss.item()

    def _save_snapshot(self, epoch):
        model = self.model
        snapshot = Snapshot(
            model_state=model.module if hasattr(model, "module") else model,
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.trainer_config.snapshot_path)
        print(f"save successfully")

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        loss = 0.0
        total_images = 0.0
        for idx, images in enumerate(dataloader):
            images = images.to(self.local_rank)
            mini_loss = self._run_batch(images, train)
            loss += mini_loss
            total_images += images.shape[0]
            print(f"Iter at  {idx} | loss {mini_loss}")

        avg_loss = loss / total_images
        print(f"Loss at {epoch}: {avg_loss}")

    def train(self):
        for epoch in range(self.epoch_run, self.trainer_config.epochs):
            epoch + 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if epoch % self.trainer_config.save_every:
                self._save_snapshot(epoch)
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
