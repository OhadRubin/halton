#!/bin/sh
#SBATCH --job-name=jobname
#SBATCH --output=out.txt        # redirect stdout
#SBATCH --error=err.txt         # redirect stderr
#SBATCH --partition=killable    # (see next section)
#SBATCH --partition=studentbatch       # (see next section)
#SBATCH --account=gpu-students
#SBATCH --time=24:00:00
#SBATCH --gpus 1
source ~/.bashrc
NODE_ID=$SLURM_ARRAY_TASK_ID bash run.sh
#usage: sbatch --array=1-30 start_job.sh
