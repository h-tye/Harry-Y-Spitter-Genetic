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

# Expected number of completed simulation txt files, directory locations
EXPECTED_FILES=@ExpectedFiles
OutFileLocation="${HOME}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log";
RunDirectoryLocation="@RunDirectoryLocation@";
DataDirectoryLocation="@DataDirectoryLocation@";
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}";
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}";
echo "OutFileLocation: ${OutFileLocation}";
echo "RunDirectoryLocation: ${RunDirectoryLocation}";
echo "DataDirectoryLocation: ${DataDirectoryLocation}";
mkdir -p $DataDirectoryLocation;
EXIT_CODE=1;

echo "####################################### Main Program: Starting #######################################";

cd $RunDirectoryLocation || exit 1;

srun  mode-solutions-batch -trust-script -logall "@name@.sbatch.lsf" >> "${OutFileLocation}" 2>&1 &
process_id=$!

exit $EXIT_CODE;

run_on_exit() {

  echo "Waiting for $EXPECTED_FILES .completed.txt files in $RunDirectoryLocation..."

  # Loop to carry out execution of sims and error handling if sims failed
  for (( i = 1; i < 20; i++ )); do

    # Submit bash file that will execute all of the individual slurm files
    echo "Running bash file to execute individual sims";
    bash run_individual_sims.sh
    EXIT_CODE=$?

    # Exit Slurm script if there are issues with bash/.lsf files
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Bash exited early with code $EXIT_CODE. Exiting Slurm job."
        exit $EXIT_CODE
    fi

    # Wait 30 seconds before checking for completed files
    sleep 30;
    LAST_MODIFIED_TIME=$(date +%s);

    # Sim marked as completed by .completed.txt file
    COMPLETED_COUNT=$(find "$RunDirectoryLocation" -maxdepth 1 -type f -name "*.completed.txt" | wc -l)
    echo "Completed files: $COMPLETED_COUNT "

    # Check if all files are generated, exit while loop if they have
    if [[ $COMPLETED_COUNT -ge $EXPECTED_FILES ]]; then
        echo "All $EXPECTED_FILES .completed.txt files detected. Proceeding to next step..."
        EXIT_CODE=0
        break
    fi

    # If sims haven't been completed in a minute, exit while loop and resubmit
    current_time=$(date +%s)
    echo "Current Time: $current_time"
    echo "Last modified time: $LAST_MODIFIED_TIME"
    time_difference=$((current_time - LAST_MODIFIED_TIME))
    if [[ $time_difference -gt 60 ]]; then
        echo "Not all sims were completed after 60 seconds, Retrying."
        # Kill the process if necessary 
        if [[ -n $process_id ]]; then
            kill -15 "$process_id" 2>/dev/null
            kill -9 "$process_id" 2>/dev/null
        fi
        EXIT_CODE=1
        break #exit loop
    fi
  
  done

  #if we have gone through the loop and our exit code is still 1 then not all files have been generated and algorithm should stop
  if [ $EXIT_CODE -eq 1 ]; then
    echo "Unsuccessful run of sims"
    exit 1
  fi

  # Carry out genetic algorithm script
  echo "Executing Genetic Algorithm"
  python "${HOME}/Y_spitter/src/genetic_alg.py"
  EXIT_CODE=$?
  # Exit Slurm script if Python script exits early
  if [ $EXIT_CODE -ne 0 ]; then
      echo "Genetic algorithm exited early with code $EXIT_CODE. Exiting Slurm job."
      exit $EXIT_CODE
  fi

  # Continue with rest of the Slurm job if needed
  echo "Python script completed successfully."

  #ensure previous directory has been destroyed first
  sleep 30;

  # find our new sim directory
  BASE_DIR="${HOME}/Y_spitter/out/lsf/"

  #All previous files cleared, search for new generated base slurm and execute
  echo "Searching for a .slurm file..."
  sleep 40;
  slurm_file=$(find "$BASE_DIR" -maxdepth 2 -type f -name "*.slurm" | head -n 1)

  if [ -z "$slurm_file" ]; then
      echo "No .slurm file found."
      exit 1  # Exit with error if no file is found
  else
      #run next iteration
      echo "Found: $slurm_file"
      sbatch "$slurm_file"
  fi

  echo "";
  echo "####################################### Billing #######################################";
  echo "";
  sacct -M "$SLURM_CLUSTER_NAME" -j "$SLURM_JOBID" --format=AllocTRES%50,elapsed;
  echo "";

  echo "";
  echo "####################################### crc-job-stats.py #######################################";
  echo "";
  crc-job-stats;
  echo "";
  echo "!!!!!! Completed !!!!!!!";
  echo "";
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "####################################### Main Program: Success #######################################";
  else
    echo "####################################### Main Program: Failure #######################################";
  fi
}
trap run_on_exit EXIT;


