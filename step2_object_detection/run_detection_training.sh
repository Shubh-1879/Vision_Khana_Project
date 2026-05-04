#PBS -N thali_detection_training
#PBS -l ncpus=16
#PBS -q gpu
#PBS -e detection_training_error.log
#PBS -o detection_training_output.log

# Change to project directory
cd $PBS_O_WORKDIR/step2_object_detection || cd ~/Vision_Khana_Project/step2_object_detection || exit 1

# Load necessary modules
module load compiler/anaconda3

# Run with unbuffered output for real-time logging
python3 -u train_object_detection.py