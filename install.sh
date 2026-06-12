#!/usr/bin/env bash
set -euo pipefail

# GPTrack environment installer.
# Recommended base environment:
#   conda create -n gptrack python=3.9 -y
#   conda activate gptrack
#   bash install.sh

python -m pip install --upgrade pip setuptools wheel

# PyTorch stack used by the reference environment.
python -m pip install \
  torch==1.12.1+cu113 \
  torchvision==0.13.1+cu113 \
  torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# Core scientific and tracking dependencies.
python -m pip install \
  numpy==1.26.4 \
  scipy==1.13.1 \
  pandas==2.2.2 \
  matplotlib==3.7.5 \
  opencv-python==4.9.0.80 \
  pillow==11.3.0 \
  pyyaml==6.0.3 \
  easydict==1.13 \
  yacs==0.1.8 \
  tqdm==4.67.1 \
  tensorboardx==2.6.2.2 \
  timm==0.9.16 \
  einops==0.7.0 \
  fvcore==0.1.5.post20221221 \
  iopath==0.1.10 \
  lmdb==1.4.1 \
  jpeg4py==0.1.4 \
  visdom==0.2.4 \
  wandb==0.16.6 \
  pycocotools==2.0.7 \
  tikzplotlib==0.10.1 \
  tabulate==0.9.0 \
  yapf==0.43.0 \
  rich==14.2.0 \
  requests==2.32.5 \
  urllib3==1.26.20

# Graph and CUDA-related packages.
python -m pip install \
  torch-geometric==2.6.1 \
  cupy-cuda12x==13.3.0

# OpenMMLab utilities. If this fails on your machine, install mmcv with
# the wheel URL matching your CUDA/PyTorch versions from OpenMMLab.
python -m pip install \
  mmengine==0.10.7 \
  mmcv==2.2.0

echo "GPTrack dependencies installed."
