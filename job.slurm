#!/bin/bash
#SBATCH --job-name=rh_parallel_test
#SBATCH --output=output.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00

module load gurobi/10.0.1
module load anaconda3
source activate conda_env

srun python main.py -ns 50 -iv original -no 100 -sm 5 -m 1
