#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH -p emmanuel
#SBATCH -t 10:00:00         # Runtime in HH:MM:SS
#SBATCH -o train_output.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e train_output.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=snarasimhamurthy@wpi.edu
#SBATCH --requeue
#SBATCH --mem 32G

# Train
python3 -u ../alcoaudio/datagen/data_processor.py --configs_file ../alcoaudio/configs/turing_configs.json