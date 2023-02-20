#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-00:01 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-4 # array size
#SBATCH -o log/testjob1/%x.%N.%A.%a.out # STDOUT  (the folder log/foldername has to exist)
#SBATCH -o log/testjob1/%x.%N.%A.%a.err # STDERR  (the folder log/foldername has to exist)
mkdir -p log/testjob1&& python3 helloworldNumber.py $SLURM_ARRAY_TASK_ID
