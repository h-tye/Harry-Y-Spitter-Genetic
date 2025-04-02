from __future__ import annotations

import argparse
import copy
import itertools
import json
from hashlib import sha256
from typing import Union, Dict, List, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd

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


# class Component:
#     instruction_syntax = "[r|]:<component_name>|*:<parameter_name>|*:[<value>|<min>:<max>:<num>]"

#     def __init__(self, sweep_str: str):
#         self.sweep_str = sweep_str

#         component = sweep_str.split(":")
#         if not (len(component) == 4 or len(component) == 6):
#             raise ValueError(
#                 f"Invalid component parameter format: {sweep_str!r} "
#                 f"(incorrect number of ':'s) (expected {self.instruction_syntax})"
#             )

#         self.types: List[str] = sorted(component[0].lower().split("|"))
#         self.names: List[str] = sorted(component[1].split("|"))
#         self.parameters: List[str] = sorted(component[2].split("|"))
#         self.value: Optional[str] = component[3] if len(component) == 4 else None
#         self.min: Optional[float] = component[3] if len(component) == 6 else None
#         self.max: Optional[float] = component[4] if len(component) == 6 else None
#         self.num: Optional[int] = int(component[5]) if len(component) == 6 else None

#         if all(i not in ["r", ""] for i in self.types):
#             raise ValueError(f"Invalid component type: {self.types!r} not in ['r', '']")

#         if "" in self.types and len(self.types) > 1:
#             raise ValueError(f"Invalid component type: {self.types!r} contains '' and other types")

#         if self.value is not None:
#             if self.min is not None or self.max is not None or self.num is not None:
#                 raise ValueError(
#                     f"Invalid component parameter format: {sweep_str!r} "
#                     f"(value given with min/max/num) (expected {self.instruction_syntax})"
#                 )
#         else:
#             if self.min is None or self.max is None or self.num is None:
#                 raise ValueError(
#                     f"Invalid component parameter format: {sweep_str!r} "
#                     f"(no value given with min/max/num) (expected {self.instruction_syntax})"
#                 )

#             self.min = float(self.min)
#             self.max = float(self.max)
#             self.num = int(self.num)

#     def __repr__(self):
#         return f"Component({self.sweep_str!r})"

#     @property
#     def short_repr(self):
#         if self.value is not None:
#             return f"{self.types}:{self.names}:{self.parameters}:{self.value}"

#         return f"{self.types}:{self.names}:{self.parameters}"

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



#main 

# .92 FOM, new model
# [[0. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0. 1. 0.]
#  [1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0.]
#  [0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 1. 1. 0. 1. 1. 0.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1.]
#  [0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1.]
#  [1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0. 1. 1. 1. 0.]
#  [0. 0. 1. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 1. 1. 1.]
#  [1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 1.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 1.]
#  [0. 1. 1. 0. 1. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 1.]
#  [0. 0. 0. 1. 1. 1. 1. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 0.]
#  [1. 0. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0.]
#  [1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 0. 0. 1.]
#  [0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 0.]
#  [0. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 1.]
#  [0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0. 1. 0. 1. 1. 0. 1. 1.]]



# Define the original "ideal" configuration as numpy array
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

for simulation_num in range(50):

    prefix_name = f"simulation_"
    # if parsed_args.ename is not None:
    #     prefix_name = prefix_name + f"{parsed_args.ename}"

    script_name = (
        f'{prefix_name}'
        f'_startup'
    )
    print(f"Script name: {script_name}")


    setup_script = SETUP_SCRIPT
    setup_script = setup_script.replace('{configuration}',format_matrix_string(hole_array))

    # generate a new configuration
    # Randomly toggle every index so we pull all truly random configs, broadens our initial search space
    for _ in range(1):
        row, col = np.random.randint(0, 20, size=2)  # Random row & column index
        hole_array.iloc[row, col] = 1 - hole_array.iloc[row, col]  # Toggle 0, 1

    common_args = dict(
        parameters=format_matrix_string(hole_array),
        setup_script=setup_script,
        script_name=script_name,
    )

    #generate individual script for configuration with uniq
    create_lsf_script(
        **common_args
    )


#first clear FOM_history file for new batch
base_directory = Path(__file__).resolve().parent.parent
history_file = base_directory / "out" / "results" / "FOM_history.txt"
with open(history_file, "w") as file:
    pass  # No need to write anything; opening in "w" mode clears the file

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

file_prefix = script_name  # No trailing backslash

# Ensure the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

#expected number of simulation files genereated, need for slurm

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


