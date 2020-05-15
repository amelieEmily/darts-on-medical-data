#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ys3316

source venv/bin/activate
python baseline.py --network resnet18