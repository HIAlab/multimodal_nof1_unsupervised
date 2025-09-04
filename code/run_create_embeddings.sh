#!/bin/bash  -eux
#SBATCH --job-name=analyse_image_nof1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=juliana.schneider@hpi.de
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --mem=32gb
#SBATCH --gpus=2
#SBATCH --exclude=hebe
#SBATCH --time=8:00:00
#SBATCH --output=/dhc/home/juliana.schneider/test_autoencoder_face/logs/%j.log # %j is job id

nvidia-smi
srun /dhc/home/juliana.schneider/conda3/envs/multimodal/bin/python ~/test_autoencoder_face/Imaging_Nof1_trial/code/Autoencoder_Analysis/Simulations_for_paper/src/create_embeddings_AE.py
