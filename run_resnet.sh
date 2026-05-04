#!/bin/bash
# Khana Dataset Classification - ResNet50 Fine-tuning Job
# Usage: qsub run_resnet.sh

#PBS -N khana_resnet_finetuning
#PBS -l ncpus=32
#PBS -q gpu
#PBS -e training_error_resnet.log
#PBS -o training_output_resnet.log

# Change to project directory
cd $PBS_O_WORKDIR || cd ~/Vision_Khana_Project || exit 1

# Load necessary modules
module load compiler/anaconda3

echo "=========================================="
echo "ResNet50 Fine-tuning Job started on $(date)"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Run with unbuffered output for real-time logging
python3 -u khana_classification_resnet.py

echo "=========================================="
echo "ResNet50 Fine-tuning Job completed on $(date)"
echo "=========================================="
