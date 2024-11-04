from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from starlette.responses import RedirectResponse
import torch
import numpy as np
from model.pointnet import ClassificationPointNet, SegmentationPointNet
from datasets import ShapeNetDataset
import random

app = FastAPI(title="Pointcloud model interactor")

@app.get("/", include_in_schema = False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Additional idea: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK"}

model_checkpoint_dir = "TM/"

def load_classification_model():
    model_checkpoint = model_checkpoint_dir + "output_cls/shapenet_classification_model.pth"
    num_classes = ShapeNetDataset.NUM_CLASSIFICATION_CLASSES
    model = ClassificationPointNet(
        num_classes=num_classes,
        point_dimension=ShapeNetDataset.POINT_DIMENSION
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def load_segmentation_model():
    model_checkpoint = model_checkpoint_dir + "output_seg/shapenet_segmentation_model.pth"
    num_classes = ShapeNetDataset.NUM_SEGMENTATION_CLASSES
    model = SegmentationPointNet(
        num_classes=num_classes,
        point_dimension=ShapeNetDataset.POINT_DIMENSION
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def point_cloud_to_json(points, colors=None):
    points_list = points.tolist()
    result = {"points": points_list}
    #if colors is not None:
    #    colors_list = colors.tolist()
    result["colors"] = colors
    return result

classification_model, classification_device = load_classification_model()
segmentation_model, segmentation_device = load_segmentation_model()

@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith('.pts'):
        raise HTTPException(status_code=400, detail="Only .pts files are accepted.")
    file_contents = await file.read()
    point_cloud_data = np.loadtxt(file_contents.decode('utf-8').splitlines()).astype(np.float32)
    points = torch.from_numpy(point_cloud_data)
    points = points.to(classification_device).unsqueeze(0)

    with torch.no_grad():
        preds, _ = classification_model(points)
        preds = preds.data.max(1)[1]
        detected_class_id = int(preds.item())

    return {"detected_class_id": detected_class_id}

@app.post("/segment")
async def segment_endpoint(file: UploadFile = File(...)):
    returnval = {
        "colors": None
    }
    if not file.filename.endswith('.pts'):
        raise HTTPException(status_code=400, detail="Only .pts files are accepted.")
    file_contents = await file.read()
    point_cloud_data = np.loadtxt(file_contents.decode('utf-8').splitlines()).astype(np.float32)
    points = torch.from_numpy(point_cloud_data)
    points = points.to(segmentation_device).unsqueeze(0)
    # Continue processing...

    with torch.no_grad():
        preds, _ = segmentation_model(points)
        preds = preds.view(-1, preds.shape[-1])
        preds = preds.data.max(1)[1]
        preds = preds.cpu().numpy()

    points_np = points.cpu().numpy().squeeze()
    num_classes = ShapeNetDataset.NUM_SEGMENTATION_CLASSES
    colors = list(range(num_classes))
    point_colors = [colors[pred] for pred in preds]
    returnval["colors"] = point_colors

    return returnval
