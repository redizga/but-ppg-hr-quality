"""1D-CNN / ResNet1D architecture for quality (task A) and HR (task B) — E4 (Budilov).

Assignment section 4: "небольшая нейросеть на сыром PPG, например 1D-CNN или
ResNet1D... подать в сеть сам временной ряд... Если используется ACC, можно
добавить отдельные каналы или отдельную ветвь сети для ACC."

This module implements only the architecture (``PPGModel``) and its shape
contract — verified here with unit tests on random tensors, no BUT PPG data
needed. The training loop itself (``scripts/train_cnn.py``: DataLoader over
the real registry, optimizer, checkpointing, best-epoch-by-validation
selection) is a separate, larger piece that depends on E1 (the registry) and
stays a stub until that data exists — building it now would be untestable
and untested code is not a real deliverable.

Design, matching ``configs/models/cnn1d.yaml``:

* ``arch: cnn1d``     -> plain conv-bn-relu stack, stride-2 downsampling.
* ``arch: resnet1d``  -> same stack with residual connections (default).
* ``channels: [ppg]`` -> PPG-only (single input branch).
* ``channels: [ppg, acc]`` -> a second, separate encoder branch for ACC
  (assignment's "отдельная ветвь сети" option — chosen over channel-stacking
  because ACC and PPG windows are not guaranteed to share the same length
  once resampling decisions land in E1; ``AdaptiveAvgPool1d`` makes both
  branches length-agnostic, so this works whether ACC ends up at 30 Hz or
  its native 100 Hz).

One architecture serves both tasks (single scalar output): task A treats it
as a logit for ``BCEWithLogitsLoss`` + sigmoid, task B treats it as the raw
HR regression value — that choice is made by the (not yet written) training
loop, not by the model.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock1D(nn.Module):
    """Conv1d -> BatchNorm1d -> ReLU, optionally stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock1D(nn.Module):
    """Basic 1D ResNet block: two convs + skip connection, stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock1D(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


def _make_encoder(arch: str, in_channels: int, base_channels: int, n_blocks: int) -> tuple[nn.Sequential, int]:
    """Build a stride-2 downsampling stack + global average pool.

    Returns ``(encoder, out_feature_dim)``. Works on any input length (the
    final ``AdaptiveAvgPool1d(1)`` makes the PPG and ACC branches
    interchangeable regardless of how long each one is).
    """
    if arch not in ("cnn1d", "resnet1d"):
        raise ValueError(f"Unknown arch {arch!r}, expected 'cnn1d' or 'resnet1d'")
    block_cls = ResidualBlock1D if arch == "resnet1d" else ConvBlock1D

    layers: list[nn.Module] = []
    channels = in_channels
    for i in range(n_blocks):
        out_channels = base_channels * (2**i)
        layers.append(block_cls(channels, out_channels, stride=2))
        channels = out_channels
    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers), channels


class PPGModel(nn.Module):
    """1D-CNN / ResNet1D over PPG, with an optional separate ACC branch.

    Input: ``ppg`` of shape ``(batch, 1, ppg_len)``; ``acc`` (if
    ``use_acc=True``) of shape ``(batch, 1, acc_len)`` — ``acc_len`` need not
    equal ``ppg_len``. Output: ``(batch,)`` — one scalar per window (a logit
    for quality, a bpm value for HR; see module docstring).
    """

    def __init__(
        self,
        arch: str = "resnet1d",
        use_acc: bool = False,
        base_channels: int = 16,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.use_acc = use_acc
        self.ppg_encoder, ppg_feat_dim = _make_encoder(arch, in_channels=1, base_channels=base_channels, n_blocks=n_blocks)

        feat_dim = ppg_feat_dim
        if use_acc:
            self.acc_encoder, acc_feat_dim = _make_encoder(
                arch, in_channels=1, base_channels=base_channels, n_blocks=n_blocks
            )
            feat_dim += acc_feat_dim

        self.head = nn.Linear(feat_dim, 1)

    def forward(self, ppg: torch.Tensor, acc: torch.Tensor | None = None) -> torch.Tensor:
        if ppg.dim() != 3 or ppg.shape[1] != 1:
            raise ValueError(f"Expected ppg shape (batch, 1, length), got {tuple(ppg.shape)}")

        feat = self.ppg_encoder(ppg).flatten(1)

        if self.use_acc:
            if acc is None:
                raise ValueError("Model was built with use_acc=True but no acc tensor was passed")
            if acc.dim() != 3 or acc.shape[1] != 1:
                raise ValueError(f"Expected acc shape (batch, 1, length), got {tuple(acc.shape)}")
            acc_feat = self.acc_encoder(acc).flatten(1)
            feat = torch.cat([feat, acc_feat], dim=1)
        elif acc is not None:
            raise ValueError("acc tensor was passed but model was built with use_acc=False")

        return self.head(feat).squeeze(-1)


def build_model_from_config(cfg: dict) -> PPGModel:
    """Build a ``PPGModel`` from a composed config (``configs/models/cnn1d.yaml``)."""
    model_cfg = cfg["model"]
    arch = model_cfg.get("arch", "resnet1d")
    channels = model_cfg.get("channels", ["ppg"])
    use_acc = "acc" in channels
    return PPGModel(arch=arch, use_acc=use_acc)
