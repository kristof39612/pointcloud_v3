import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# Import ShapeNetDataset and model classes
from datasets import ShapeNetDataset
from model.pointnet import ClassificationPointNet, SegmentationPointNet

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score

def evaluate(model, dataloader, device, task):
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = model(inputs)

        if task == 'classification':
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            accuracy = (preds == targets).mean()
            precision = precision_score(targets, preds, average='weighted')
            recall = recall_score(targets, preds, average='weighted')
            f1 = f1_score(targets, preds, average='weighted')
            conf_matrix = confusion_matrix(targets, preds)

            print(f'Classification Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'Confusion Matrix:\n{conf_matrix}')

        elif task == 'segmentation':
            preds = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1)
            preds = preds.data.max(1)[1]

            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            accuracy = (preds == targets).mean()
            iou = jaccard_score(targets, preds, average='weighted')
            dice = f1_score(targets, preds, average='weighted')

            print(f'Segmentation Accuracy: {accuracy:.4f}')
            print(f'Mean IoU: {iou:.4f}')
            print(f'Dice Coefficient: {dice:.4f}')
        else:
            raise ValueError('Unknown task!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation'], required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--number_of_points', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    test_dataset = ShapeNetDataset(
        dataset_folder=args.dataset_folder,
        number_of_points=args.number_of_points,
        task=args.task,
        train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    # Calculate class weights for balancing
    if(args.task == 'classification'):
        class_counts = {}
        for _, target in test_dataset:
            category = target.item() if args.task == 'classification' else target.view(-1).max().item()
            if category in class_counts:
                class_counts[category] += 1
            else:
                class_counts[category] = 1
    
        class_weights = {category: 1.0 / count for category, count in class_counts.items()}
        sample_weights = [class_weights[target.item() if args.task == 'classification' else target.view(-1).max().item()] for _, target in test_dataset]
    
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers
        )

    # Load model
    if args.task == 'classification':
        model = ClassificationPointNet(
            num_classes=test_dataset.NUM_CLASSIFICATION_CLASSES,
            point_dimension=test_dataset.POINT_DIMENSION
        )
    elif args.task == 'segmentation':
        model = SegmentationPointNet(
            num_classes=test_dataset.NUM_SEGMENTATION_CLASSES,
            point_dimension=test_dataset.POINT_DIMENSION
        )
    else:
        raise ValueError('Unknown task!')

    model.load_state_dict(torch.load(args.model_checkpoint,weights_only=True))
    model.to(device)

    # Evaluate model
    evaluate(model, test_loader, device, args.task)