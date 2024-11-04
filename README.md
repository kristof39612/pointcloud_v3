# Point Cloud Processing with PyTorch PointNet

This repository contains the NHF project for point cloud processing and visualization.
It includes various tools and algorithms for handling 3D point cloud data, enabling 
efficient manipulation, analysis, and rendering of large-scale point cloud datasets.
**NHF repo for point cloud magic.**

## Authors

- **Lovácsi Kristóf**
    - Neptun: N3EEWB
    - Username: kristof39612

- **Dremák Gergely**
    - Neptun: KSHSLY
    - Username: wingsmc

## Getting Started

### Prerequisites

- Python 3.11+ 
- PyTorch
- Docker (optional, for containerized execution)

### Python Package Requirements

| Package      | Description                                                                 | Source |
|--------------|-----------------------------------------------------------------------------|--------|
| torch        | *PyTorch is used for building and training deep learning models.*             | [Link](https://pytorch.org/get-started/locally/) |
| numpy        | *NumPy is used for numerical operations and handling arrays.*                 | [Link](https://pypi.org/project/numpy/) |
| fastprogress | *FastProgress is used for displaying progress bars during training.*          | [Link](https://pypi.org/project/fastprogress/) |
| datasets     | *Datasets library is used for handling and processing datasets.*              | [Link](https://pypi.org/project/datasets/) |
| uvicorn      | *Uvicorn is a lightning-fast ASGI server implementation, using `uvloop` and `httptools`.* | [Link](https://pypi.org/project/uvicorn/) |
| matplotlib   | *Matplotlib is used for creating static, animated, and interactive visualizations.* | [Link](https://pypi.org/project/matplotlib/) |
| open3d       | *Open3D is an open-source library that supports rapid development of software that deals with 3D data.* | [Link](http://www.open3d.org/) |

*Note: The package requirements are stashed in a file called requirements.txt*
## Installation

A more optimized Docker container will be available in the future that trains the model and saves it to a specified directory. This container will streamline the training process, ensuring that all dependencies and configurations are correctly set up.

*Notes:*
- The docker container has the Shapenet_partanno dataset embedded and unpacked to a directory.
- Due to Shapenet regulations the Shapenet_v0 dataset is unavailable in the official site, even with registration. The dataset is stored with **GIT LFS in the repo. Make sure you have GIT LFS enabled. (The repo has subscription purchased for bandwidth)**
- Further progress on this Dockerfile depends on actions taken in the repo and may be modified as the solution progresses.

### Docker containerization
#### Building the Docker container for training (cls + seg)
```sh
docker build -t pointcloud/train:latest .
```
#### Running the Docker Container
To run the Docker container, use:
```sh
docker run -v <OUTPUT_FOLDER>:/trained_model pointcloud/train:latest
```

---

#### Building the Docker container for evaluation
```sh
docker build -t pointcloud/eval:latest -f Dockerfile.eval .
```
#### Running the Docker Container for evaluation
To run the Docker container for evaluation, use:
```sh
docker run -v <OUTPUT_FOLDER>:/trained_model --gpus all pointcloud/eval:latest
```
---
#### Building the Docker container for Internal Inference API
```sh
docker build -t pointcloud/api:latest -f Dockerfile.infer .
```
#### Running the Docker Container for Internal Inference API
To run the Docker container for evaluation, use:
```sh
docker run -v <OUTPUT_FOLDER>:/trained_model --gpus all pointcloud/api:latest
```

---

If you want to use GPU-s with the Docker container pass along the ```--gpus all``` flag before the image name.

*Note: **replace <OUTPUT_FOLDER> parameter** with your desired local folder and include ${PWD} if necessary.*

## Docker compose
Provided in the submodule for the frontend. **For more details, refer to the [Frontend README](point_cloud_client/README.md).**

## Local setup

Create a Python virtual environment and install the package requirements before progressing further!

Linux commands:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Note: If you want to run the Internal API you must install ```uvicorn gunicorn fastapi python-multipart``` too.*

```sh
pip install uvicorn gunicorn fastapi python-multipart
```

### Training the Model locally

To train the classification model locally, run:
```sh
python train.py shapenet shapenet_partanno_v0 classification output_cls --number_of_workers 4 --epoch 15
```
To train the segmentation model locally, run:
```sh
python train.py shapenet shapenet_partanno_v0 segmentation output_seg --number_of_workers 4 --epoch 15
```
*Note: Replace **output_cls** and **output_seg** directories with the correct folder you desire.*

---

### Infering with models

A front-end with a graphical user interface (GUI) will be provided. This front-end will interact with the models via an API, allowing users to easily test and visualize the results of classification and segmentation tasks.

#### Via our interface
Our inference runs on the web front end included in the submodule, but it requires the internal API docker image to be up and running with a trained model mounted to its `/trained_model` path before doing anything.

**For more details, refer to the [Frontend README](point_cloud_client/README.md).**

---

#### Via Open3D (Command line call)

For the inference script locally with the **trained classification** model, run:
```sh
python infer.py shapenet output_cls/shapenet_classification_model.pth ../point_clouds_to_test/lampa2.pts classification
```

For the inference script locally with the **trained segmentation** model, run:
```sh
python infer.py shapenet output_seg/shapenet_segmentation_model.pth ../point_clouds_to_test/lampa2.pts segmentation
```
*Note: Replace **output_cls** and **output_seg** directories with the correct folder that contains the .pth file for the corresponding model.*

*Note: Replace the **../point_clouds_to_test/lampa2.pts** with the desired .pts file you want.*

## Related Works

| Type | Title | Authors | Link |
|------|-------|---------|------|
| Paper | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas | [Link](https://arxiv.org/abs/1612.00593) |
| Paper | PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space | Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas | [Link](https://arxiv.org/abs/1706.02413) |
| Paper | Dynamic Graph CNN for Learning on Point Clouds | Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon | [Link](https://arxiv.org/abs/1801.07829) |
| Paper | PointCNN: Convolution On X-Transformed Points | Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen | [Link](https://arxiv.org/abs/1801.07791) |
| Repository | PointNet2 Repository | Charles R. Qi | [Link](https://github.com/charlesq34/pointnet2) |
| Dataset | Shapenet | Various | [Link](https://shapenet.org/) |

## Contact
For questions or support, please contact us at [lovacsi.kristof@gmail.com](mailto:lovacsi.kristof@gmail.com).