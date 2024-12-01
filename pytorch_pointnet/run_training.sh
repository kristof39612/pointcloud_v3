#!/bin/bash

# Run classification training
echo "Starting Classification training..."
python train.py shapenet shapenet_partanno_v0_final classification /trained_model/cls --number_of_workers 4 --epoch 25
echo "Classification training finished!"
echo "-----------------------------------"
echo "-----------------------------------"
echo "Starting Segmentation training..."
# Run segmentation training
python train.py shapenet shapenet_partanno_v0_final segmentation /trained_model/seg --number_of_workers 4 --epoch 25
echo "Segmentation training finished!"