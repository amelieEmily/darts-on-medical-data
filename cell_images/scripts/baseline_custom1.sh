#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
source /vol/bitbucket/ys3316/darts-on-medical-data/cell_images/venv/bin/activate
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
python baseline.py --network custom1