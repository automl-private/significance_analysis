#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 4000 # memory pool for all cores (4GB)
#SBATCH -t 0-00:01 # time (D-HH:MM)
#SBATCH -c 1 # number of cores
#SBATCH -a 1-4 # array size
#SBATCH -o log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID



mkdir -p ./example/jobtest &&  python3 helloworldNumber.py $SLURM_ARRAY_TASK_ID &&

if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
   echo Testecho
   exit $?
fi
