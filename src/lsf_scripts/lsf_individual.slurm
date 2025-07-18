#!/bin/bash
#SBATCH --job-name=@name@
#SBATCH --output=slurm_%x_%A.log
#SBATCH --mail-user=hit24@pitt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --time=1-00:00:00
#SBATCH --chdir="/ihome/nyoungblood/hit24"
#SBATCH --requeue

source "${HOME}/.bashrc";
module load intel intel-mpi lumerical;

echo "####################################### Main Program: Starting #######################################";

cd $RunDirectoryLocation || exit 1;

srun  varfdtd-engine-mpich2nem -trust-script -logall "@individual_sim_name@.run.lms"
sleep 30; 
srun mode-solutions-batch -trust-script -logall "@load_sim_name@.run.lsf"
process_id=$!