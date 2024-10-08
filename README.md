# Point Cloud Processing with PyTorch PointNet

NHF repo for point cloud magic.

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch
- Docker (optional, for containerized execution)

### Installation

Dockerfile is present but it copies a dummy file (train_sample.py). 
- **It doesn't train a model currently**.
- The docker container has the Shapenet_partanno dataset embedded and unpacked to a directory.
- Further progress on this Dockerfile depends on actions taken in the repo and may be modified as the task progresses.

### Building the Docker image for training

```sh
docker build .
```

### Training the Model

TBA

*For testing CLS*:
```sh
python train.py shapenet shapenet_partanno_v0_final  classification output_cls --number_of_workers 4 --epoch 15
```
*For testing SEG*:
```sh
python train.py shapenet shapenet_partanno_v0_final  segmentation output_seg --number_of_workers 4 --epoch 15
```

### Infering with models

TBA

*For testing CLS*: 
```sh
python infer.py shapenet output_cls/shapenet_classification_model.pth ../point_clouds_to_test/lampa2.pts classification
```

*For testing SEG*: 
```sh
python infer.py shapenet output_seg/shapenet_segmentation_model.pth ../point_clouds_to_test/lampa2.pts segmentation
```