#!/bin/bash
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=chipakupoy@gmail.com
#SBATCH --mail-type=ALL

echo "Starting the job at $(date)"

source /home2/anika.roy/miniconda3/bin/activate ML
python3 -u ppi.py

echo "Job finished at $(date)"