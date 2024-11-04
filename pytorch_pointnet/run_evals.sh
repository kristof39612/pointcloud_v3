#!/bin/bash

# Run classification eval
echo "Starting Classification eval..."
python eval_trained.py --dataset_folder shapenet_partanno_v0_final --task classification --model_checkpoint /trained_model/output_cls/shapenet_classification_model.pth --batch_size 128
echo "Classification eval finished!"
echo "-----------------------------------"
echo "-----------------------------------"
echo "Starting Segmentation eval..."
python eval_trained.py --dataset_folder shapenet_partanno_v0_final --task segmentation --model_checkpoint /trained_model/output_seg/shapenet_segmentation_model.pth --batch_size 128
echo "Segmentation eval finished!"