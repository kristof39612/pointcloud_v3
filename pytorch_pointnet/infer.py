import argparse
import random
import json
import numpy as np

import open3d.visualization
import torch

import open3d

from model.pointnet import ClassificationPointNet, SegmentationPointNet
from datasets import ShapeNetDataset, PointMNISTDataset

MODELS = {
    'classification': ClassificationPointNet,
    'segmentation': SegmentationPointNet
}

DATASETS = {
    'shapenet': ShapeNetDataset,
    'mnist': PointMNISTDataset
}

def point_cloud_to_json(pcd):
    """
    Converts a point cloud object to a JSON string.
    Args:
        pcd: A point cloud object that contains points, and optionally colors and normals.
    Returns:
        A JSON string representation of the point cloud. The JSON object contains:
            - "points": A list of points, where each point is represented as a list of coordinates.
            - "colors" (optional): A list of colors corresponding to the points, where each color is represented as a list of RGB values.
            - "normals" (optional): A list of normals corresponding to the points, where each normal is represented as a list of normal vector components.
    """
    points = list(map(lambda x: list(x), pcd.points))
    
    point_cloud_dict = {
        "points": points
    }
    
    if pcd.has_colors():
        colors = list(map(lambda x: list(x), pcd.colors))
        point_cloud_dict["colors"] = colors
    
    if pcd.has_normals():
        normals = list(map(lambda x: list(x), pcd.normals))
        point_cloud_dict["normals"] = normals
    
    return json.dumps(point_cloud_dict, indent=4)

def infer(dataset, model_checkpoint, point_cloud_file, task):
    """
    Perform inference on a point cloud using a pre-trained model.
    Args:
        dataset (str): The name of the dataset to use.
        model_checkpoint (str): Path to the model checkpoint file.
        point_cloud_file (str): Path to the point cloud file to be inferred.
        task (str): The task to perform, either 'classification' or 'segmentation'.
    This function loads a pre-trained model, prepares the point cloud data, 
    performs inference, and visualizes the results. For classification tasks, 
    it prints the detected class and visualizes the point cloud. For segmentation 
    tasks, it colors the points based on the predicted classes and visualizes the 
    colored point cloud. The point cloud data is also saved to a JSON file.
    """
    if task == 'classification':
        num_classes = DATASETS[dataset].NUM_CLASSIFICATION_CLASSES
    elif task == 'segmentation':
        num_classes = DATASETS[dataset].NUM_SEGMENTATION_CLASSES
    model = MODELS[task](num_classes=num_classes,
                         point_dimension=DATASETS[dataset].POINT_DIMENSION)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_checkpoint,weights_only=True))

    points = DATASETS[dataset].prepare_data(point_cloud_file)
    points = torch.tensor(points)
    if torch.cuda.is_available():
        points = points.cuda()
    points = points.unsqueeze(dim=0)
    model = model.eval()
    preds, feature_transform = model(points)
    if task == 'segmentation':
        preds = preds.view(-1, num_classes)
    preds = preds.data.max(1)[1]

    points = points.cpu().numpy().squeeze()
    preds = preds.cpu().numpy()

    if task == 'classification':
        print('Detected class: %s' % preds)
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((49,1))])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        open3d.visualization.draw_geometries([pcd])
    elif task == 'segmentation':
        colors = [(random.randrange(256)/255, random.randrange(256)/255, random.randrange(256)/255)
                  for _ in range(num_classes)]
        rgb = [colors[p] for p in preds]
        rgb = np.array(rgb)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(rgb)
        open3d.visualization.draw_geometries([pcd])

    json_data = point_cloud_to_json(pcd)

    with open("point_cloud.json", "w") as json_file:
        json_file.write(json_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['shapenet', 'mnist'], type=str, help='dataset to train on')
    parser.add_argument('model_checkpoint', type=str, help='dataset to train on')
    parser.add_argument('point_cloud_file', type=str, help='path to the point cloud file')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')

    args = parser.parse_args()

    infer(args.dataset,
          args.model_checkpoint,
          args.point_cloud_file,
          args.task)
