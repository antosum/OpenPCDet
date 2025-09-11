# OpenPCDet Project Context & Agent Guide

## Project Overview

**OpenPCDet** is a comprehensive PyTorch-based codebase for LiDAR-based 3D object detection, developed by OpenMMLab. It serves as the official implementation for multiple state-of-the-art 3D detection methods and supports various autonomous driving datasets.

### Core Capabilities
- **Multiple SOTA Models**: PointRCNN, Part-A2-Net, PV-RCNN, Voxel R-CNN, PV-RCNN++, MPPNet, BEVFusion, CenterPoint
- **Multi-Dataset Support**: KITTI, NuScenes, Waymo, Lyft, ONCE, Argoverse2, Pandaset, **Custom Datasets**
- **Multi-Modal Detection**: LiDAR-camera fusion models (BEVFusion, TransFusion)
- **Temporal Detection**: Multi-frame detection capabilities
- **Distributed Training**: Multi-GPU/multi-machine support

## User's Enhancements & Contributions

### 1. Custom Dataset Infrastructure
**Primary Focus**: Productionizing 3D object detection for custom autonomous driving datasets

#### Key Files Created:
- `tools/create_geminai_infos.py` - Generates info files for Geminai dataset
- `tools/validate_and_make_imagesets.py` - Validates dataset files and creates train/val splits
- `tools/compute_point_cloud_range.py` - Automatic voxel parameter estimation for custom datasets
- `tools/merge_custom_datasets.py` - Merges multiple custom datasets with proper indexing
- `tools/fine_tuning_plan.md` - Comprehensive fine-tuning strategies

#### Enhanced Files:
- `pcdet/datasets/custom/custom_dataset.py` - Added 4D point cloud support with timestamp padding
- `tools/cfgs/dataset_configs/geminai_dataset.yaml` - Geminai dataset configuration

### 2. Advanced Training System
**Progressive Unfreezing & Multi-stage Training**

#### Key Files Modified:
- `tools/train.py` - Enhanced with WANDB integration, periodic evaluation, configurable workers
- `tools/train_utils/train_utils.py` - Added periodic evaluation logic, training optimizations
- `tools/train_utils/optimization/` - Enhanced optimization strategies for progressive unfreezing

#### Configuration Files:
- `tools/cfgs/geminai_models/` - Complete multi-stage training configs:
  - `geminai_cbgs_dyn_pp_centerpoint_stageA.yaml`
  - `geminai_cbgs_dyn_pp_centerpoint_stageB.yaml` 
  - `geminai_cbgs_dyn_pp_centerpoint_stageC.yaml`
  - `geminai_cbgs_dyn_pp_centerpoint.yaml` (main config)

### 3. WANDB Integration
**Complete Experiment Tracking System**

#### Key Files:
- `tools/train_utils/wandb_utils.py` - WANDB utility functions
- Enhanced `tools/train.py` and `tools/test.py` with WANDB logging
- Updated YAML configs with WANDB parameters and tags

### 4. Infrastructure Improvements
- `pcdet/datasets/__init__.py` - Added conditional imports for Argo2Dataset and WaymoDataset
- `pcdet/models/backbones_2d/base_bev_backbone.py` - Fixed np.int compatibility for PyTorch 2.0
- `smoke.py` - CUDA ops verification and environment details script
- `docs/finetune_dyn_pillars_custom.md` - Enhanced fine-tuning documentation

## Project Structure & Key Files

### Core Library (`pcdet/`)
```
pcdet/
├── datasets/                    # Dataset implementations
│   ├── custom/                 # Custom dataset support
│   │   └── custom_dataset.py  # **ENHANCED**: Custom dataset with 4D support
│   ├── kitti/, nuscenes/, waymo/, etc.  # Standard datasets
│   └── processor/              # Data processing pipelines
├── models/                     # Model implementations
│   ├── detectors/              # Main detector models
│   ├── backbones_3d/           # 3D backbone networks
│   ├── dense_heads/            # Detection heads
│   └── ops/                    # CUDA operations
├── utils/                      # Utility functions
└── config.py                   # Configuration management
```

### Tools & Scripts (`tools/`)
```
tools/
├── cfgs/                       # Configuration files
│   ├── dataset_configs/         # Dataset configurations
│   │   ├── geminai_dataset.yaml  # **NEW**: Geminai dataset config
│   │   └── custom_dataset.yaml
│   ├── geminai_models/         # **NEW**: Geminai model configs
│   ├── kitti_models/, nuscenes_models/, etc.
│   └── custom_models/
├── train.py                    # **ENHANCED**: Main training script with WANDB
├── test.py                     # **ENHANCED**: Testing script with WANDB
├── eval_utils/eval_utils.py    # **ENHANCED**: Evaluation utilities
├── train_utils/                # Training utilities
│   ├── train_utils.py          # **ENHANCED**: Training logic with periodic eval
│   ├── optimization/           # **ENHANCED**: Optimization strategies
│   └── wandb_utils.py          # **NEW**: WANDB integration
├── create_geminai_infos.py     # **NEW**: Geminai dataset info generation
├── validate_and_make_imagesets.py  # **NEW**: Dataset validation
├── compute_point_cloud_range.py   # **NEW**: Voxel parameter estimation
├── merge_custom_datasets.py    # **NEW**: Dataset merging tool
├── fine_tuning_plan.md         # **NEW**: Fine-tuning strategies
└── smoke.py                    # **NEW**: Environment verification
```

### Documentation (`docs/`)
```
docs/
├── finetune_dyn_pillars_custom.md  # **ENHANCED**: Fine-tuning guide
├── CUSTOM_DATASET_TUTORIAL.md
├── GETTING_STARTED.md
├── INSTALL.md
└── [other documentation files]
```

## Key Configuration Files

### Training Configs (User's Focus)
- `tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint.yaml` - Main Geminai CenterPoint config
- `tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint_stage[A|B|C].yaml` - Multi-stage training configs

### Dataset Configs
- `tools/cfgs/dataset_configs/geminai_dataset.yaml` - Geminai dataset configuration
- `tools/cfgs/dataset_configs/custom_dataset.yaml` - Generic custom dataset template

## Important Commands & Workflows

### Dataset Preparation
```bash
# Generate Geminai dataset info files
python tools/create_geminai_infos.py

# Validate dataset and create train/val splits
python tools/validate_and_make_imagesets.py

# Compute point cloud range for custom datasets
python tools/compute_point_cloud_range.py

# Merge multiple custom datasets
python tools/merge_custom_datasets.py
```

### Training with User's Enhancements
```bash
# Train with WANDB integration and periodic evaluation
python tools/train.py --cfg_file tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint.yaml

# Multi-stage progressive unfreezing training
python tools/train.py --cfg_file tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint_stageA.yaml
python tools/train.py --cfg_file tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint_stageB.yaml
python tools/train.py --cfg_file tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint_stageC.yaml
```

### Testing & Evaluation
```bash
# Test with WANDB logging
python tools/test.py --cfg_file tools/cfgs/geminai_models/geminai_cbgs_dyn_pp_centerpoint.yaml

# Environment verification
python smoke.py
```

## User's Technical Focus Areas

1. **Production Deployment**: Custom dataset support for real-world autonomous driving
2. **Training Optimization**: Progressive unfreezing, multi-stage training, periodic evaluation
3. **Experiment Tracking**: Comprehensive WANDB integration for reproducible research
4. **Infrastructure**: Robust dataset handling, environment verification, compatibility fixes

## Key Dependencies & Environment
- **PyTorch**: 1.1-1.10 (with PyTorch 2.0 compatibility fixes)
- **spconv**: Sparse convolution library
- **WANDB**: Experiment tracking (newly integrated)
- **CUDA**: Custom operations for 3D computations
- **Standard ML stack**: NumPy, OpenCV, tensorboardX, etc.

This guide provides a comprehensive overview of the OpenPCDet project context with emphasis on the user's specific enhancements and contributions, particularly focused on custom dataset support, advanced training methodologies, and experiment tracking infrastructure.