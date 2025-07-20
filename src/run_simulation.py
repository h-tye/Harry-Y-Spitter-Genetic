from pathlib import Path
import sys
import numpy as np
import pandas as pd
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)  # Adds the parent directory
from out import get_lsf_path
from out import get_results_path
from src.lsf_scripts import get_lsf_scripts_path


# Load the simulation setup script from an external file
SETUP_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "src" / "lsf_scripts" / "individual_script_template.lsf"
LOAD_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "src" / "lsf_scripts" / "load_script_template.lsf"
INDIVIDUAL_SLURM_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "src" / "lsf_scripts" / "lsf_individual.slurm"
LOAD_SCRIPT = LOAD_SCRIPT_PATH.read_text(encoding="utf-8")
SETUP_SCRIPT = SETUP_SCRIPT_PATH.read_text(encoding="utf-8")
INDIVIDUAL_SLURM_SCRIPT = INDIVIDUAL_SLURM_SCRIPT_PATH.read_text(encoding="utf-8")

'''
Format a DataFrame as a string representation of a matrix.
'''
def format_matrix_string(df: pd.DataFrame) -> str:
    
    matrix_str = "[ "  # Start the matrix string

    for i, row in df.iterrows():
        row_str = ", ".join(map(str, row))  # Convert row values to comma-separated string
        matrix_str += row_str  # Append row string
        if i < len(df) - 1:
            matrix_str += ";\n              "  # Add semicolon and spacing for new row

    matrix_str += " ]"  # Close the matrix string
    return matrix_str

'''
Create an LSF script for a given configuration.
'''
def create_child_lsf_script(configuration, script_name,script_dir):

    # Define the path to the LSF script in the script directory
    lsf_script_path = Path(script_dir) / f"{script_name}.run.lsf"

    # Replace placeholders in the template with actual values
    lsf_script = SETUP_SCRIPT.replace("{configuration}", configuration)

    # Write the LSF script to the file
    lsf_script_path.write_text(lsf_script, encoding="utf-8")

'''
Create a load LSF script to execute the simulation.
'''
def create_load_lsf_script(script_name,script_dir):

    lms_filename = f"{script_name}.lms"

    # Define the path to the load script
    load_script_path = Path(script_dir) / f"{script_name}.load.lsf"

    load_script = LOAD_SCRIPT.replace("{lms_filename}", lms_filename)

    # Write the load script to the file
    load_script_path.write_text(load_script, encoding="utf-8")

'''
Create an LSF script for each configuration in the initial generation.
'''
def create_individual_slurm_script(script_name,script_dir):

    slurm_filename = f"{script_name}.slurm"

    # Define the path to the individual slurm script
    individual_slurm_script_path = Path(script_dir) / slurm_filename

    slurm_script = INDIVIDUAL_SLURM_SCRIPT.replace("@individual_sim_name@",script_name)
    slurm_script = slurm_script.replace("@load_sim_name@", script_name)

    # Write the individual slurm script to the file
    individual_slurm_script_path.write_text(slurm_script, encoding="utf-8")

    return slurm_filename


def create_initial_generation(num_children,mutation_intensity,script_name,script_dir):

    # Define our starting configuration
    hole_array = pd.DataFrame([
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,],
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1,],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1,],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,],
        [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,],
        [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0,],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1,],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1,]
    ])


    # Create bash file that will run all the individual slurm files
    bash_script = "#!/bin/bash\n"


    # Generate 50 child configurations from the starting configuration
    for sim_idx in range(num_children):

        print(f"Script name: {script_name}")

        # Create a copy of the original array so we don't modify our starting point
        hole_array_temp = hole_array.copy()

        # Generate a new configuration by toggling 1 pixel in our starting configuration
        for _ in range(mutation_intensity):
            row, col = np.random.randint(0, 20, size=2)  # Random row & column index
            hole_array_temp.iloc[row, col] = 1 - hole_array.iloc[row, col]  # Toggle 0, 1

        # Generate individual script for configuration with uniq
        create_child_lsf_script(format_matrix_string(hole_array_temp),f"{script_name}_{sim_idx}", script_dir)

        slurm_filename = create_individual_slurm_script(f"{script_name}_{sim_idx}",script_dir)

        bash_script += f"sbatch {slurm_filename}\n"

        create_load_lsf_script(f"{script_name}_{sim_idx}",script_dir)

    # Write the bash script to a file
    bash_script_path = script_dir / "run_all.sh"
    bash_script_path.write_text(bash_script, encoding="utf-8")


def main(script_name="simulation_startup",generation_size=50, mutation_intensity=1):

    # First clear FOM_history file for new batch
    base_directory = Path(__file__).resolve().parent.parent
    history_file = base_directory / "out" / "results" / "FOM_history.txt"
    with open(history_file, "w") as file:
        pass  # No need to write anything; opening in "w" mode clears the file

    # Generate slurm file for all scripts, code pulled from lsf.py 
    location = get_lsf_path()
    data_location = get_results_path()

    # Create directory based on the script name
    script_dir = location / script_name
    script_dir.mkdir(parents=True, exist_ok=True)

    create_initial_generation(
        generation_size,  # Number of child configurations to generate
        mutation_intensity,  # Number of pixels to toggle in each configuration
        script_name,
        script_dir
    )

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

