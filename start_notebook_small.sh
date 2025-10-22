#!/bin/bash
#SBATCH --partition med2 
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6 
#SBATCH --mem=12G
#SBATCH --time 12:00:00
#SBATCH --job-name jupyter-high
#SBATCH --output jupyter-notebook-high.log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=salba@ucdavis.edu=

# get tunneling info
# create random port to avoid conflicts with other people on node
port=$(shuf -i8000-9999 -n1)

# get node/user/cluster info
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print}')

# set localport to 8888 unless otherwise specified
if [[ -z "${jupyter_port}" ]]; then
	localport=8888
else
	localport=${jupyter_port}
fi

module load slurm julia
unset XDG_RUNTIME_DIR


# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${localport}:${cluster}:${port} ${user}@farm.cse.ucdavis.edu

For more info and how to connect from windows,
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port: ${localport}
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${localport}  (prefix w/ https:// if using password)
"

# This part actually runs the notebook
jupyter-lab --no-browser --port=${port} --ip=${node}
