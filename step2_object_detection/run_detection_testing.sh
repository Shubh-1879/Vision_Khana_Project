#PBS -N thali_detection_training
#PBS -l ncpus=16
#PBS -q gpu
#PBS -e detection_testing_error.log
#PBS -o detection_testing_output.log

# Change to project directory
cd $PBS_O_WORKDIR || exit 1

# Load necessary modules
module load compiler/anaconda3

# Run with unbuffered output for real-time logging
python3 -u test_object_detection.py
