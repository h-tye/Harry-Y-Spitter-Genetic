#File structure:
# 1. Pull data from results file
# 2. Check FOM
# 3. If FOM is good for any design, return and cleanup
# 4. Check FOM with previous best FOMs to make sure loss is decreasing
# 5. Run genetic component to get hole matrices
# 6. Using these matrices, generate 20-25 new files, repeat

import os
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os
import shutil
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)  # Adds the parent directory

from out import get_output_path
from out import get_lsf_path
from out import get_results_path
from src.lsf_scripts import get_lsf_scripts_path
from src.run_simulation import create_child_lsf_script, create_load_lsf_script, create_individual_slurm_script, format_matrix_string

class Configuration:
    def __init__(self, hole_matrix, fom,filename):
        self.hole_matrix = pd.DataFrame(hole_matrix)
        self.fom = fom
        self.filename = filename

def cleanup(directory, move_to_dir=None):

    # Create the move_to directory if it doesn't exist
    if move_to_dir and not os.path.exists(move_to_dir):
        os.makedirs(move_to_dir)
    
    # First pass: Move .run.txt files to seperate directory to use later
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if file has .run.txt extension and we have a move directory
            if move_to_dir and filename.endswith('.run.txt'):
                dest_path = os.path.join(move_to_dir, filename)
                shutil.move(file_path, dest_path)
                print(f"Moved: {file_path} -> {dest_path}")
        except Exception as e:
            print(f"Failed to move {file_path}. Reason: {e}")
    
    # Second pass: Delete all remaining files, do a lot to ensure deletion
    for _ in range(700): 
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def read_in_txt_matrix(file_path):
    with open(file_path, "r") as file:
        # Read the single line from the file
        line = file.readline().strip()
        
        # Split the line into rows using semicolons
        rows = line.split(";")
        
        # Split each row into columns using tabs and convert to integers
        matrix = [
            [int(x) for x in row.strip().split("\t") if x.strip()]
            for row in rows if row.strip()
        ]
    
    # Convert the matrix to a pandas DataFrame
    return pd.DataFrame(matrix)


def read_in_txt_fom(file_path):
    try:
        with open(file_path, "r") as file:
            return float(file.readlines()[-1].strip())
    except Exception:
        return 0.50

def sort_by_fom(configs):
    return sorted(configs, key=lambda config: config.fom, reverse=True)

def get_last_fom_values(history_file, num_values):
    if not Path(history_file).exists():
        return []
    with open(history_file, "r") as file:
        lines = file.readlines()
    return [float(line.strip()) for line in lines[-num_values:]]

def write_fom_to_history(history_file, fom):
    with open(history_file, "a") as file:
        file.write(f"{fom}\n")

def long_term_LMR(top_five, previous_foms,gamma):

    # get an idea of how much we have changed recently(relative to last 4 iterations)
    total_difference = top_five[0].fom - previous_foms[-1] #calculate current gradient
    for j in range(3):
        difference = previous_foms[-j-1] - previous_foms[-j-2]
        total_difference = total_difference + difference
    
    avg_grad = total_difference / 4

    #then figure out how this compares in a broader sense so that we can normalize
    global_total = 0
    for k in range(15):
        global_difference = abs(previous_foms[-k] - previous_foms[-k-1])
        global_total = global_total + global_difference
    normalized_grad = global_total / 15

    #define our coefficient to alter mutation strength. Squared so it is extra sensitive to large increases
    improvement_rate = (avg_grad * avg_grad) / (normalized_grad * normalized_grad)

    print(f"Current Iteration Best: {top_five[0].fom}")
    print(f"Previous Best: {previous_foms[0]}")
    print(f"Average change over the last 4: {avg_grad}")
    print(f"Average change over the last 15: {normalized_grad}")
    print(f"Improvemt Rate: {improvement_rate}")

    # initiate extinction event if neccessary
    if len(previous_foms) > 100 and abs(normalized_grad) < .004:
        mutation_strength = 25
        print(f"Have not converged to a solution and minimal change in FOM, initiating extinction...")
        return mutation_strength

    #set LMR
    mutation_strength = 1

    #if we are moving up, large increases means slow down and investigate sorroundings, hence 
    # it is inversely proportional to gradient. In the case that the average gradient is small
    # it is still likely to be N(.000064), {N<10} so mutation strength should remain reasonable
    if avg_grad > 0:
        mutation_strength = int(mutation_strength * (1/(improvement_rate+gamma))) + 1

    #if we are moving down, large decreases indicates we are exiting an optimal search space
    #and should thus start increasing our mutation rate to find a new one. If the decreases are small,
    # we could be finely navigating an area and should thus slow our mutation rate to comb over search space
    else :
        mutation_strength = int(abs(improvement_rate*gamma) * mutation_strength) + 1

    

    return mutation_strength

def calc_beta(current_fom, best_fom, previous_foms):
    
    #if we are not in long run, keep beta as 1
    if len(previous_foms) < 100:
        return 1
    else :

        #first figure out what typical distance from the best is
        global_total = 0
        for k in range(15):
            global_difference = abs(previous_foms[-k] - best_fom)
            global_total = global_total + global_difference
        normalized_grad = global_total / 15

        beta = 1 + int((abs(current_fom - best_fom) / normalized_grad))
        return beta

def create_generation(top_five,num_child_configs, mutation_strength, script_name, script_dir):
    # Create bash file that will run all the individual slurm files
    bash_script = "#!/bin/bash\n"

    for config in top_five:

        hole_array = config.hole_matrix


        # Generate child configurations from the best configurations
        for sim_idx in range(num_child_configs):
            
            # Create a copy of the original array so we don't modify our starting point
            hole_array_temp = hole_array.copy()

            # Generate a new configuration by toggling 1 pixel in our starting configuration
            for _ in range(mutation_strength):
                row, col = np.random.randint(0, 20, size=2)  # Random row & column index
                hole_array_temp.iloc[row, col] = 1 - hole_array.iloc[row, col]  # Toggle 0, 1

            # Generate individual script for configuration with uniq
            create_child_lsf_script(format_matrix_string(hole_array_temp),f"{script_name}_{sim_idx}", script_dir)

            slurm_filename = create_individual_slurm_script(f"{script_name}_{sim_idx}",script_dir)

            bash_script += f"sbatch {slurm_filename}\n"

            create_load_lsf_script(f"{script_name}_{sim_idx}",script_dir)

    # Write the bash script to a file
    bash_script_path = Path(script_dir) / f"{script_name}_run.sh"
    bash_script_path.write_text(bash_script, encoding="utf-8")




def main(script_name="simulation_gen",generation_size=50):

    # Directory and history file
    # Find the correct results directory
    base_directory = Path(__file__).resolve().parent.parent  # Navigate to Y_splitter/
    history_file = base_directory / "out" / "results" / "FOM_history.txt"
    array_history = base_directory / "out" / "results" / "Array_history.txt"
    file_history = base_directory / "out" / "results" / "File_history.txt"
    base_directory = base_directory / "out" / "lsf"  # Navigate to Y_splitter/out/lsf
    results_directory = next(
        (d for d in base_directory.iterdir() 
        if d.is_dir() and d.name.startswith("simulation") and any(d.iterdir())), 
        None
    )


    if results_directory is None:
        raise FileNotFoundError("No simulation results directory found in Y_splitter/out/lsf/")

    iteration_configs = []

    # Read configurations from result files
    for file in results_directory.iterdir():
        full_suffix = "." + file.name.split(".", 1)[-1]
        if full_suffix == ".run.txt":
            hole_matrix = read_in_txt_matrix(file)
            fom = read_in_txt_fom(file)
            filename = file
            iteration_configs.append(Configuration(hole_matrix, fom,filename))

    # Sort by FOM value in descending order
    top_five = sort_by_fom(iteration_configs)[:5]

    # Ensure loss function is decreasing
    i = 1
    previous_foms = get_last_fom_values(history_file, 500)

    #hyperparameters to decide ramge of genetic variance
    alpha = 12 #for first equation, coefficient to bias what range of mutation what
    gamma = (1/2) #for second equation, 1/gamma = max possible mutation strength
    mutation_strength = alpha  #for first run

    # Calculate mutation strength based on previous FOMs
    while previous_foms and i <= len(previous_foms):
        previous_fom = previous_foms[-i]  # Check the most recent FOM values
        if top_five[0].fom >= .99:
            write_fom_to_history(history_file, top_five[0].fom)
            write_fom_to_history(file_history, top_five[0].filename)
            write_fom_to_history(array_history, top_five[0].hole_matrix)
            print("FOM exceeds .99")
            exit()
        if i == 1:

            #calculate mutation strength:
            mutation_strength = int((1 - ((top_five[0].fom - previous_fom) / previous_fom) * alpha)) + 1

            print(f"Mutation Strength: {mutation_strength}")
            break
        
        if top_five[0] > previous_fom:
            break #break out if our fom is greater than any of the last 5

        #if loss function is not decreasing, break
        if i >= 5:
            print("Loss has not decreased past 5 iterations, Restarting from current best")
            cleanup(results_directory)
            exit()
        i += 1

    # safety stop condition
    if len(previous_foms) > 200:
        with open(history_file, "a") as file:
            file.write(f"More than 200 iterations. Stopping...")
        exit()

    # Write the best FOM to history and cleanup
    write_fom_to_history(history_file, top_five[0].fom)
    write_fom_to_history(file_history, top_five[0].filename)
    database_location =  base_directory / "data_storage"
    cleanup(results_directory, database_location)

    # Generate slurm file for all scripts, code pulled from lsf.py 
    location = get_lsf_path()
    data_location = get_results_path()

    # Create directory based on the script name
    script_dir = location / script_name
    script_dir.mkdir(parents=True, exist_ok=True)

    create_generation(top_five,10,mutation_strength, script_name, script_dir)


    # Generate slurm and lsf scripts for the simulation
    slurm_lsf = get_lsf_scripts_path().joinpath("lsf.slurm").read_text(encoding="utf-8")
    slurm_lsf = slurm_lsf.replace("@name@", f"{script_name}")
    slurm_lsf = slurm_lsf.replace("@ExpectedFiles", f"{generation_size}")
    slurm_lsf = slurm_lsf.replace("@RunDirectoryLocation@", str(script_dir))
    slurm_lsf = slurm_lsf.replace("@DataDirectoryLocation@", str(data_location))
    script_dir.joinpath(f"{script_name}.lsf.slurm").write_text(slurm_lsf, encoding="utf-8")

    lsf_script = get_lsf_scripts_path().joinpath("sbatch.lsf").read_text(encoding="utf-8")
    lsf_script = lsf_script.replace("@name@", f"{script_name}")
    lsf_script = lsf_script.replace("@num_files", f"{generation_size}")
    lsf_script = lsf_script.replace("@RunDirectoryLocation@", str(script_dir))
    lsf_script = lsf_script.replace("@DataDirectoryLocation@", str(data_location))
    script_dir.joinpath(f"{script_name}.sbatch.lsf").write_text(lsf_script, encoding="utf-8")

main()