#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --account=csnlp_jobs
#SBATCH --output=%j.out
#SBATCH --gpus=1

source /work/courses/3dv/23-2/ben/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/csnlp/Team1/envs/csnlp-ben

python train_nce.py