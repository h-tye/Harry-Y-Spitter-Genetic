#File structure:
# 1. Pull data from txt file
# 2. Check FOM
# 3. If FOM is good for any design, return and cleanup
# 4. Check FOM with previous best FOMs to make sure loss is decreasing
# 5. Run genetic component to get hole matrices
# 6. Using these matrices, generate 20-25 new files, repeat

# Psuedo
# class configuration(
#     private:
#         holeMatrix;
#         FOM;
# )

# vector<configuration> iteration_configs

# for( files in results_directory){
#     if(last4(filename) == ".txt"){

#         configuation current_config;
#         configuartion.holeMatrix = read_in_txt_matrix(file);
#         configuration.FOM = read_in_txt_fom(file);
#         iteration_configs.push_back(current_config);

        
#     }
# }
    

# sort_by_fom(iteration_configs);
# vector<configuration> top_five;
# for(i in 5){
#     top_five.push_back(iteration_configs(i));
# }

# # psuedo

# #ensure loss function decreasing
# do{
#     previous_FOM = read("FOM_history", last_line - i);
#     if(i > 5){
#         print("Loss has not decreased past 5 iterations");
#         exit;
#     }
#     i++;
# }while(previous_FOM < top_five[:1].FOM);

# write("FOM_history", top_five[:1].FOM);

import os
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import shutil
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)  # Adds the parent directory

from out import get_output_path
from src.functions.__const__ import HASH_LENGTH
from src.functions.lsf_script import create_lsf_script
from src.functions.lsf_script import create_lsf_script_sweep
from src.functions.run_sim import run_sweep
from out import get_lsf_path
from out import get_results_path
from src.compile_data import get_compile_data_path
from src.functions.param_to_combinations import param_to_combinations
from src.functions.process_scripts import process_scripts
from src.lsf_scripts import get_lsf_scripts_path

class Configuration:
    def __init__(self, hole_matrix, fom):
        self.hole_matrix = pd.DataFrame(hole_matrix)
        self.fom = fom

def cleanup(directory):
    #destroy all files in directory, but keeep directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):  # Check if it's a file or symbolic link
                os.unlink(file_path)  # Delete the file or symbolic link
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
    with open(file_path, "r") as file:
        return float(file.readlines()[-1].strip())

def sort_by_fom(configs):
    return sorted(configs, key=lambda config: config.fom, reverse=True)

def get_last_fom_values(history_file, num_values=5):
    if not Path(history_file).exists():
        return []
    with open(history_file, "r") as file:
        lines = file.readlines()
    return [float(line.strip()) for line in lines[-num_values:]]

def write_fom_to_history(history_file, fom):
    with open(history_file, "a") as file:
        file.write(f"{fom}\n")

def format_matrix_string(df: pd.DataFrame) -> str:
    #Convert a Pandas DataFrame into a formatted matrix string.
    matrix_str = "[ "  # Start the matrix string

    for i, row in df.iterrows():
        row_str = ", ".join(map(str, row))  # Convert row values to comma-separated string
        matrix_str += row_str  # Append row string
        if i < len(df) - 1:
            matrix_str += ";\n              "  # Add semicolon and spacing for new row

    matrix_str += " ]"  # Close the matrix string
    return matrix_str

# Directory and history file
# Find the correct results directory
base_directory = Path(__file__).resolve().parent.parent  # Navigate to Y_splitter/
history_file = base_directory / "out" / "results" / "FOM_history.txt"
base_directory = base_directory / "out" / "lsf"  # Navigate to Y_splitter/out/lsf
results_directory = next((d for d in base_directory.iterdir() if d.is_dir() and d.name.startswith("simulation")), None)

if results_directory is None:
    raise FileNotFoundError("No simulation results directory found in Y_splitter/out/lsf/")

iteration_configs = []

# Read configurations from result files
for file in results_directory.iterdir():
    full_suffix = "." + file.name.split(".", 1)[-1]
    if full_suffix == ".run.txt":
        hole_matrix = read_in_txt_matrix(file)
        fom = read_in_txt_fom(file)
        iteration_configs.append(Configuration(hole_matrix, fom))

# Sort by FOM value in descending order
top_five = sort_by_fom(iteration_configs)[:5]


# Ensure loss function is decreasing
i = 1
previous_foms = get_last_fom_values(history_file, num_values=5)


#stopping conditions, if fom is good continue, if fom is really good or loss is bad exit
while previous_foms and i <= len(previous_foms):
    previous_fom = previous_foms[-i]  # Check the most recent FOM values
    if top_five[0].fom >= .70:
        print("FOM exceeds .70")
        cleanup(results_directory)
        exit()
    if top_five[0].fom > previous_fom:
        break
    if i >= 5:
        print("Loss has not decreased past 5 iterations")
        cleanup(results_directory)
        exit()
    i += 1

# safety stop condition
if len(previous_foms) > 2:
    with open(history_file, "a") as file:
        file.write(f"More than 2 iterations. Stopping...")
    exit()

# Write the best FOM to history
print(top_five[0].fom)
write_fom_to_history(history_file, top_five[0].fom)

#clear directory we've been in running to make room for new iteration
cleanup(results_directory)


SETUP_SCRIPT = r'''
    deleteall;
    clear;
    switchtolayout;


    #base simulation, this is a model of what "should" be created by our run_simulation script

    #define grid size dimensions
    grid_size_x = 20;
    grid_size_y = 20;
    cell_width = 120e-9;
    cell_height = 120e-9;

    si_sqr_xspan = 2.4e-6;
    si_sqr_yspan = 2.4e-6;
    wg_w = 0.5e-6;

    #define some stuff for FOM calculations
    FOM = 0;
    a = 2 / 3;

    #add solver
    addvarfdtd;
    set('mesh accuracy', 6);
    set("x", 0);
    set('x span', 3.5e-6);
    set("y", 0);
    set('y span', 3.5e-6);
    set("z max", 1.5e-6);
    set('z min', -1.5e-6);
    set('y0', -(si_sqr_xspan / 2 + 0.2e-6));
    set("x0", -(si_sqr_yspan - wg_w) / 2);

    #add substrate
    addrect;
    set('name', 'BOX');
    set("x", 0);
    set("y", 0);
    set('x span', 5e-6);
    set('y span', 5e-6);
    set("z max", -0.11e-6);
    set('z min', -2e-6);
    set('material', "SiO2 (Glass) - Palik");

    ######################################

    #add rectangle for silicon
    addrect();
    set("name", "si_sqr");
    set("x", 0);
    set("y", 0);
    set("z", 0);
    set('x span', si_sqr_xspan);
    set('y span', si_sqr_yspan);
    set('z span', 220e-9);
    set('material', 'Si (Silicon) - Palik');

    addrect();
    set("name", "wg_out_cross");
    set("x min", si_sqr_xspan / 2);
    set('x max', si_sqr_xspan / 2 + 1.5e-6);
    set("y", (si_sqr_yspan - wg_w) / 2);
    set('y span', wg_w);
    set("z", 0);
    set('z span', 220e-9);
    set('material', 'Si (Silicon) - Palik');

    addrect();
    set("name", "wg_out_thru");
    set("y min", si_sqr_xspan / 2);
    set('y max', si_sqr_xspan / 2 + 1.5e-6);
    set("x", -(si_sqr_yspan - wg_w) / 2);
    set('x span', wg_w);
    set("z", 0);
    set('z span', 220e-9);
    set('material', 'Si (Silicon) - Palik');

    addrect();
    set("name", "wg_in_thru");
    set("y max", -si_sqr_xspan / 2);
    set('y min', -(si_sqr_xspan / 2 + 1.5e-6));
    set("x", -(si_sqr_yspan - wg_w) / 2);
    set('x span', wg_w);
    set("z", 0);
    set('z span', 220e-9);
    set('material', 'Si (Silicon) - Palik');

    ######################################

    #add source
    addmodesource;
    set("injection axis", "y");
    set('set wavelength', 1);
    set("wavelength start", 1.5e-6);
    set("wavelength stop", 1.6e-6);
    set("x", -(si_sqr_yspan - wg_w) / 2);
    set('y', -(si_sqr_xspan / 2 + 0.2e-6));
    set("x span", 2.5e-6);

    #add monitor
    addpower;
    set("name", "mon_top_down");
    set("x", 0);
    set("y", 0);
    set('x span', si_sqr_xspan + 0.5e-6);
    set('y span', si_sqr_yspan + 0.5e-6);
    set('z', 0);

    #add index monitor(big monitor)
    addindex;
    set("name", "mon_index");
    set("x", 0);
    set("y", 0);
    set('x span', si_sqr_xspan + 1.5e-6);
    set('y span', si_sqr_yspan + 1.5e-6);
    set('z', 0);

    #add transmittance monitors
    addpower;
    set("name", "mon_source");
    set("monitor type", "2D Y-normal");
    set("x", -(si_sqr_yspan - wg_w) / 2);
    set('y', -(si_sqr_xspan / 2 + 0.1e-6));
    set('z', 0);
    set('x span', 1e-6);
    set('z span', 1e-6);

    addpower;
    set("name", "mon_thru");
    set("monitor type", "2D Y-normal");
    set("x", -(si_sqr_yspan - wg_w) / 2);
    set('y', (si_sqr_xspan / 2 + 0.4e-6));
    set('z', 0);
    set('x span', 1e-6);
    set('z span', 1e-6);

    addpower;
    set("name", "mon_cross");
    set("monitor type", "2D X-normal");
    set('x', (si_sqr_xspan / 2 + 0.4e-6));
    set("y", (si_sqr_yspan - wg_w) / 2);
    set('z', 0);
    set('y span', 1e-6);
    set('z span', 1e-6);

    #add changes

    addstructuregroup;
    set('name', 'hole_array');

    #define srating hole pattern
    holeArray = {configuration};
    holeMatrix = "";

    for (i = 1:length(holeArray(:,1))) {  # Loop over rows
        row_str = num2str(holeArray(i, :));  # Convert row to string
        holeMatrix = holeMatrix + row_str + "; ";  # Append row with separator
    }


    #select and delete old cirles
    select('hole_array');
    delete;

    #create circles based on starting array
    for (i = 0; i < grid_size_x; i = i + 1)
    {
        for (j = 0; j < grid_size_y; j = j + 1)
        {
    #calculate the position for each cell
            x_position = (i)*cell_width - si_sqr_xspan / 2;
            y_position = (j)*cell_height - si_sqr_xspan / 2;

    #create each cell
            addcircle();
            set('name', 'Circle');
            set('radius', 45e-9);
            set('x', x_position + cell_width / 2);
            set('y', y_position + cell_height / 2);
            set('z span', 222e-9);
            addtogroup('hole_array');

            if (holeArray(i + 1, j + 1) == 0)
            {
                set('material', 'Si (Silicon) - Palik');
            }
            else
            {
                set('material', 'etch');
            }
        }
    }
    

'''

i = 1 #iterator marker
iteration_num = len(previous_foms) #number of iterations is marked by number of foms stored to history
for configuration in top_five:
    #now we just use run_simulation.py within a for loop

    #pull hole array from what we are currently
    hole_array = configuration.hole_matrix

    #for each configuration, generate 4 child configs
    for simulation_num in range(4):

        #script name = iteration + accuracy marker(1-5) + individual simulation_num
        script_name = (
            f'simulation_'
            f'_iteration_num'
            f'{iteration_num}_'
            f'{i}_'
            f'{simulation_num}'
        )
        print(f"Script name: {script_name}")

        setup_script = SETUP_SCRIPT
        setup_script = setup_script.replace('{configuration}',format_matrix_string(hole_array))

        common_args = dict(
            parameters=format_matrix_string(hole_array),
            setup_script=setup_script,
            script_name=script_name,
        )

        #generate individual script for configuration with uniq
        create_lsf_script(
            **common_args
        )

        # generate a new configuration
        # Randomly indices, num of indices is proportional to the FOM, lower FOM => more random
        for _ in range(5*i):
            row, col = np.random.randint(0, 20, size=2)  # Random row & column index
            hole_array.iloc[row, col] = 1 - hole_array.iloc[row, col]  # Toggle 0, 1

#generate slurm file for all scripts, code pulled from lsf.py 
location = get_lsf_path()
data_location = get_results_path()

location = location.joinpath(script_name).absolute()
data_location = data_location.joinpath(script_name).absolute()

location.mkdir(exist_ok=True)

#move lsf scripts we created into directory
# Construct paths correctly
source_path = Path(__file__).resolve().parent.parent / "out" / "lsf"
destination_path = source_path / script_name  # Avoid string concatenation

source_folder = Path(source_path)
destination_folder = Path(destination_path)

file_prefix = (
            f'simulation_'
            f'_iteration_num'
            f'{iteration_num}_'  
        )# No trailing backslash

# Ensure the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

expected_files = 0
# Move files with correct glob pattern
for file in source_folder.glob(file_prefix + "*"):  # Matches files starting with script_name
    if file.is_file():  
        file.rename(destination_folder / file.name)  
        print(f"Moved: {file} → {destination_folder}")
        expected_files = expected_files + 1
    else:
        print(f"Skipping directory: {file}")

print(f'{location}')
data_location.mkdir(exist_ok=True)
print(f'{data_location}')
location_str = str(location).replace("\\", "/")
data_location_str = str(data_location).replace("\\", "/")
compile_data_py_str = str(get_compile_data_path()).replace("\\", "/")

slurm_lsf = get_lsf_scripts_path().joinpath("lsf.slurm").read_text(encoding="utf-8")
slurm_lsf = slurm_lsf.replace("@name@", f"{script_name}")
slurm_lsf = slurm_lsf.replace("@ExpectedFiles",f"{expected_files}")
slurm_lsf = slurm_lsf.replace("@RunDirectoryLocation@", location_str)
slurm_lsf = slurm_lsf.replace("@DataDirectoryLocation@", data_location_str)
location.joinpath(f"{script_name}.lsf.slurm").write_text(slurm_lsf, encoding="utf-8")

lsf_script = get_lsf_scripts_path().joinpath("sbatch.lsf").read_text(encoding="utf-8")
lsf_script = lsf_script.replace("@name@", f"{script_name}")
lsf_script = lsf_script.replace("@RunDirectoryLocation@", location_str)
lsf_script = lsf_script.replace("@DataDirectoryLocation@", data_location_str)
lsf_script = lsf_script.replace("@compile_data_py@", compile_data_py_str)
location.joinpath(f"{script_name}.sbatch.lsf").write_text(lsf_script, encoding="utf-8")