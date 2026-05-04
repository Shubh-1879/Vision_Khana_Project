#!/bin/bash

#PBS -N khana_classification
#PBS -l ncpus=32
#PBS -q gpu
#PBS -e training_error.log
#PBS -o training_output.log

# Change to project directory
cd $PBS_O_WORKDIR || cd ~/Vision_Khana_Project || exit 1

# Load necessary modules
module load compiler/anaconda3

# Run with unbuffered output for real-time logging
python3 -u khana_classification.py
