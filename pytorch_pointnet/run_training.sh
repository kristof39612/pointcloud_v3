#!/bin/bash

# Run classification training
echo "Starting Classification training..."
python train.py shapenet shapenet_partanno_v0 classification /trained_model/cls --number_of_workers 4 --epoch 15
echo "Classification training finished!"
echo "-----------------------------------"
echo "-----------------------------------"
echo "Starting Segmentation training..."
# Run segmentation training
python train.py shapenet shapenet_partanno_v0 segmentation /trained_model/seg --number_of_workers 4 --epoch 15
echo "Segmentation training finished!"