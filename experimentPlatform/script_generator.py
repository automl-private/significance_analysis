import os

arraySize = 4
partition = "testdlc_gpu-rtx2080"
memory = 4000
scriptIndex = len(os.listdir("./experimentPlatform/clusterScripts"))
foldername = "testjob1"

# Create the Bash script
# Prepare Experiment
bash_script = "#!/bin/bash\n"
bash_script += "#SBATCH -p " + partition + " # partition (queue)\n"
bash_script += "#SBATCH --mem " + str(memory) + " # memory pool for all cores (4GB)\n"
bash_script += "#SBATCH -t 0-00:01 # time (D-HH:MM)\n"
bash_script += "#SBATCH -c 1 # number of cores\n"
bash_script += "#SBATCH -a 1-" + str(arraySize) + " # array size\n"
# bash_script += "mkdir -p log/%x.%N.%A"
bash_script += (
    "#SBATCH -o log/"
    + foldername
    + "/%x.%N.%A.%a.out # STDOUT  (the folder log/foldername has to exist)\n"
)
bash_script += (
    "#SBATCH -o log/"
    + foldername
    + "/%x.%N.%A.%a.err # STDERR  (the folder log/foldername has to exist)\n"
)


# Execute Experiment
bash_script += (
    "mkdir -p log/" + foldername + "&& python3 helloworldNumber.py $SLURM_ARRAY_TASK_ID"
)


# Write the Bash script to a file
with open(
    "./experimentPlatform/clusterScripts/clusterScript" + str(scriptIndex) + ".sh", "w"
) as f:
    f.write(bash_script)
