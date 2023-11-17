#!/bin/bash
#SBATCH -n 10
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=chipakupoy@gmail.com
#SBATCH --mail-type=ALL

source activate ML
python3 scalability.py