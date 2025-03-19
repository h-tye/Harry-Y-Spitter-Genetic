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

# Expected number of completed simulation txt files
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

run_on_exit() {

  echo "Waiting for $EXPECTED_FILES .completed.txt files in $RunDirectoryLocation..."

  #run multiple iterations to ensure files are created
  for (( i = 1; i < 4; i++ )); do
    #If directory is good, now run generated sims
    echo "Running bash file to execute individual sims";
    bash run_individual_sims.sh
    EXIT_CODE=$?

    # Exit Slurm script if there are issues with bash/.lsf files
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Bash exited early with code $EXIT_CODE. Exiting Slurm job."
        exit $EXIT_CODE
    fi

    LAST_MODIFIED_TIME=$(date +%s);
    # Wait until all completed files are found
    while true; do
        # Count the number of ".completed.txt" files using `find` (more robust than `ls`)
        COMPLETED_COUNT=$(find "$RunDirectoryLocation" -maxdepth 1 -type f -name "*.completed.txt" | wc -l)
        echo "Completed files: $COMPLETED_COUNT "

        # Check if all files are generated
        if [[ $COMPLETED_COUNT -ge $EXPECTED_FILES ]]; then
            echo "All $EXPECTED_FILES .completed.txt files detected. Proceeding to next step..."
            EXIT_CODE=0
            break
        fi

        # Check for timeout
        current_time=$(date +%s)
        echo "Current Time: $current_time"
        echo "Last modified time: $LAST_MODIFIED_TIME"
        time_difference=$((current_time - LAST_MODIFIED_TIME))
        if [[ $time_difference -gt 100 ]]; then
            echo "Not all sims were completed after 100 seconds, Retrying."
            # Kill the process if necessary (ensure $process_id is defined elsewhere)
            if [[ -n $process_id ]]; then
                kill -15 "$process_id" 2>/dev/null
                kill -9 "$process_id" 2>/dev/null
            fi
            EXIT_CODE=1
            break #exit loop
        fi

        # Wait for the specified interval before checking again
        sleep 30
    done

    if [ $EXIT_CODE -eq 0 ]; then
      break;
    fi
  
  done

  #if we have gone through 4 iterations and all failed, exit
  if [ $EXIT_CODE -eq 1 ]; then
    echo "Unsuccessful run of sims"
    exit 1
  fi

  echo "Executing Genetic Algorithm"
  
  python "${HOME}/Documents/Y_spitter/src/genetic_alg.py"
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
  BASE_DIR="${HOME}/Documents/Y_spitter/out/lsf/"
  # SIM_DIR=$(ls -d "$BASE_DIR"simulation__iteration* 2>/dev/null)

  # # Check if the directory was found
  # if [[ -z "$SIM_DIR" ]]; then
  #     echo "Error: No directory starting with 'simulation' found in $BASE_DIR. Sim dir: $SIM_DIR"
  #     exit 1
  # fi

  echo "Simulation directory found: $BASE_DIR"

  #All previous files cleared, search for new generated base slurm and execute
  for (( i = 1; i < 4; i++ )); do
    echo "Searching for a .slurm file..."
    slurm_file=$(find "$BASE_DIR" -maxdepth 2 -type f -name "*.slurm" | head -n 1)
    sleep 20;
  done

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

echo "####################################### Main Program: Starting #######################################";

cd $RunDirectoryLocation || exit 1;










EXIT_CODE=1;
LAST_MODIFIED_TIME=$(date +%s);
for (( i = 1; i < 2; i++ )); do
  echo "${i}th attempt";
  srun  mode-solutions-batch -trust-script -logall "@name@.sbatch.lsf" >> "${OutFileLocation}" 2>&1 &
  process_id=$!

  while ps -p $process_id > /dev/null; do
    if [ -f "$SLURM_JOB_ID.completed.txt" ]; then
      EXIT_CODE=0;
      rm "$SLURM_JOB_ID.completed.txt";
      echo "Completed Successfully";
      kill -15 $process_id;
      kill -9 $process_id;
      break;
    fi

    if [ -f "$SLURM_JOB_ID.run.txt" ]; then
      LAST_MODIFIED_TIME=$(stat -c %Y "$SLURM_JOB_ID.run.txt");
      cat "$SLURM_JOB_ID.run.txt";
      rm "$SLURM_JOB_ID.run.txt";
    fi

    current_time=$(date +%s);
    time_difference=$((current_time - LAST_MODIFIED_TIME));
    if [ $time_difference -gt 600 ]; then
      echo "No progress in the 10 minutes. Killing the process";
      kill -15 $process_id;
      kill -9 $process_id;
      break;
    fi
    
    sleep 10;
  done

  if [ $EXIT_CODE -eq 0 ]; then
    break;
  fi

  sleep 5;
done

all_files=$(ls ./*."${SLURM_JOB_ID}".running.lsf);
for file in $all_files; do
  file_without_num=$file;
  file_without_num=${file_without_num%.*};
  file_without_num=${file_without_num%.*};
  file_without_num=${file_without_num%.*};
  new_file=${file_without_num}.run.lsf;
  echo "$file -> $new_file";
  mv "$file" "$new_file";
done

exit $EXIT_CODE;
