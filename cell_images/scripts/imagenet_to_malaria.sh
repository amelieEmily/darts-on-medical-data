#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
source /vol/bitbucket/ys3316/darts-on-medical-data/cell_images/venv/bin/activate
TERM=vt100
/usr/bin/nvidia-smi
uptime
python imagenet_to_malaria.py --dataset malaria --search_wd 0.0009 --dearch_dp 0.0--auxiliary --model_path imagenet_model.pt