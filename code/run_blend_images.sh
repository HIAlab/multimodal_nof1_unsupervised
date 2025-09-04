#!/bin/bash  -eux
#SBATCH --job-name=blend_images
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juliana.schneider@hpi.de
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --mem=16gb
#SBATCH --gpus=2
#SBATCH --time=8:00:00
#SBATCH --output=/dhc/home/juliana.schneider/test_autoencoder_face/logs/study_analysis/%j.log # %j is job id

nvidia-smi
srun /dhc/home/juliana.schneider/conda3/envs/multimodal/bin/python ~/Nof1/Analyse_Multimodal_Nof1/test_autoencoder_face/Imaging_Nof1_trial/code/Autoencoder_Analysis/Simulations_for_paper/create_blended_images.py
