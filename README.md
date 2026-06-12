
## Highlights

- **Geometry-Saliency Cross-Attention (GSCA):** Integrates imaging-inspired guidance into cross-modal interaction.
- **Multi-Relation Topology Graph (MRTG):** Builds spatial, cross-modal, and semantic edges for graph-enhanced RGB-T representation learning.
- **LasHeR-ready Training and Evaluation:** Provides experiment configs and scripts for RGB-T tracking workflows.

## Repository Structure

```text
GPTrack/
├── experiments/gptrack/        # Training and evaluation configs
├── lib/
│   ├── config/gptrack/         # Default GPTrack configuration
│   ├── models/gptrack/         # GPTrack model and backbone modules
│   ├── train/                  # Training actors, datasets, and trainers
│   └── test/                   # Evaluation and tracker wrappers
├── pretrained_models/          # Pretrained weights
└── tracking/                   # Training, testing, and analysis entry points
```

## Installation

Create the environment and install the required dependencies:

```bash
conda create -n gptrack python=3.8
conda activate gptrack
bash install.sh
```

## Project Path Setup

Initialize local paths for datasets, checkpoints, and results:

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After initialization, paths can be adjusted in:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

## Data Preparation

Place RGB-T datasets under `./data`. For LasHeR, the expected structure is:

```text
data/
└── lasher/
    ├── trainingset/
    ├── testingset/
    ├── trainingsetList.txt
    └── testingsetList.txt
```

## Pretrained Weights

Download the SOT pretrained weights and place them in:

```text
pretrained_models/
```


## Training

Train GPTrack on LasHeR with the SOT initialization config:

```bash
python tracking/train.py \
  --script gptrack \
  --config vitb_256_gptrack_32x1_1e4_lasher_15ep_sot \
  --save_dir ./output/vitb_256_gptrack_32x1_1e4_lasher_15ep_sot \
  --mode multiple \
  --nproc_per_node 4
```

Available configs are located in:

```text
experiments/gptrack/
```

## Evaluation

Run tracking on the LasHeR test split:

```bash
python tracking/test.py \
  gptrack \
  vitb_256_gptrack_32x1_1e4_lasher_15ep_sot \
  --dataset_name lasher_test \
  --threads 6 \
  --num_gpus 1
```

Analyze tracking results:

```bash
python tracking/analysis_results.py \
  --tracker_name gptrack \
  --tracker_param vitb_256_gptrack_32x1_1e4_lasher_15ep_sot \
  --dataset_name lasher_test
```



## Model Components

The core GPTrack implementation is organized around:

- `lib/models/gptrack/gptrack.py`: tracker wrapper and model builder.
- `lib/models/gptrack/vit_gptrack_backbone.py`: ViT backbone with GSCA and MRTG modules.
- `lib/models/gptrack/utils.py`: token conversion utilities and multi-relational edge construction.


## Citation

If GPTrack is useful for your research, please cite the related work and this repository.
