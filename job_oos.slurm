#!/bin/bash
#SBATCH --job-name=disaster_model
#SBATCH --output=output.log
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G

module load gurobi/10.0.1
module load anaconda3
source activate conda_env

srun python main.py -ns 2 -iv original -sm 1 -eos -oos 2

