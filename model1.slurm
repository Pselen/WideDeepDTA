#!/bin/bash
#SBATCH -A tbag80 # your_user_account
#SBATCH -J dta_1
#SBATCH -N 1 #number of nodes to be used
#SBATCH -n 1 # number of tasks (mpi) to be lunched
#SBATCH -c 20 #number of cores per task. Not mandatory
#SBATCH -p akya-cuda #name of the job queue
#SBATCH --gres=gpu:1
#SBATCH --time=03:15:00
#SBATCH --output=/truba/home/sparlar/WideDeepDTA/logs/dta_1.log
#SBATCH --mail-user=parlarselen@gmail.com
#SBATCH --mail-type=ALL

~/anaconda3/envs/deepdta/bin/python -u runner_model1.py