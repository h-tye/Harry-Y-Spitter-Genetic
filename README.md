# Overview
The goal of this project is to inversely design a waveguide splitter to conform to any ratio at the output. This is primarily focused on a 2/3 output at the through port and a 1/3 output at the cross port. To navigate to an optimal solution, we use a genetic algorithm to navigate the search space of possible configurations and locate optimal solutions. Work consulted: "Arbitrary-Direction, multichannel and ultra-compact power splitters by inverse design method” – Ma et. Al. Simulations of design were done via Ansys Lumerical and the University of Pittsburgh Computer Resource Center. 

# Implementation
The waveguide splitter is designed to be implemented on a 2.4 uM x 2.4 uM silicon crystal. In order to test different geometric configurations of this design, we parse the structure into a 20 x 20 grid(i.e. 400 "pixel" structure). Each pixel can be completely etched and contain only air, or can be filled with Silicon(Si). We injected a source of light with wavelengths between 1.5 uM and 1.6 uM into the configuration and then measured the average output across a bandwidth of 500 nm. A figure-of-merit(FOM) was used to judge how much the configuration differed from the ideal output by the following equation: 
FOM = 1 - (abs(T_thru_avg - T_thru_ideal) + abs(T_cross_avg - T_cross_ideal))

![image](https://github.com/user-attachments/assets/bc733dd6-7166-4dd8-9721-12d38ff252da)


A genetic algorithm based off the FOM was then used to selectively search for the best configuration.

# Genetic Algorithm
The basic pipeline of the current genetic algorithm as follows:
  1. Generate 50 random configurations
  2. Run sims on configurations
  3. Retrieve the 5 best configurations(based by FOM)
  4. Generate 9 children for each of the best 5
     
     a. Children are made by randomly toggling a certain number of pixels within the parent structure. Hence we are introducing "mutations" to the children.
     
     b. The best configuration will have the lowest mutation rate(LMR) and we gradually increase the mutation rate for the next 4. For example, if our LMR is 1
     then the best configuration will generate 9 children that have 1 randomly mutated pixel. The second best configuartion will generate 9 children that have 2
     mutated pixels, etc.
  6. We then generate 5 more moderately random sims(mutation rate of 25) to ensure genetic variance within the generation.
  7. This then comprises the next generation where we will run steps 2-4 until a desirable FOM is achieved or some other terminal condition is reached.
This pipeline is implemented within /src/genetic_alg.py

# Lowest Mutation Rate(LMR)
The amount of genetic variance we introduce into a generation is determined by the lowest mutation rate(LMR). This parameter is extremely crucial in determining convergence to optimal solutions with the search space. Since we are using a 20 x 20 binary grid, we can expect a search space of 2^400 possible configurations. From what I have seen, the search space is extremely sensitive and a large LMR is likely to skip over high FOM regions. For context, a generation of 50 sims that only differed by 1 pixel produced a FOM range of around 15%. For this reason, I have designed the LMR to remain under 3 for most contexts.

Another feature to consider is to have an adaptive LMR. Due to the reasons mentioned above, it would be risky to keep the LMR at 3+ if we have just made a huge improvement in our FOM. Instead, we want to slow down the mutation rate so that we can comb over the local search space and look for any extrema. Additionally, we don't want to keep the LMR small if our FOM has not improved over the past 5 generations. Thus the LMR should be a function of the FOM such that large increases correspond to low LMR and vice versa. 

The current LMR equation is as shown:

LMR = floor(1 - ((current_generation_best - previous_generation_best) / previous_generation_best) * alpha ) + 1

Alpha is hyperparameter that configures LMR senstivity to recent changes. We add 1 to the result to ensure that LMR is never 0. 
This LMR has provided the fastest convergence compared to other methods such as constant LMR and momementum-based LMR. 
**LMR is/should be subject to change**

# Beam Splitter Nueral Network(BSNN)
BSNN initially was an attempt to train a CNN to recognize optimal features within our configurations and then generate a configuartion of high accuracy(FOM>.99). It was trained on 40K test configurations and acheived low loss within the test data. However, when run in a sim, around 20% loss was seen and thus was not an accurate enough model to carry out its function. 
That being said, when a BSNN generated configuration was used as a starting configuration to generate the first generation(as opposed to randomly generating children), convergence to high FOM regions(FOM > .90) was achieved in 20 or less generations. This is opposed to our original method which took 100+ generations to reach the same level of FOM. Additionally, when a non-BSNN generated configuration with similar FOM was used as a starting position, the FOM degenerated for 10+ generations before increasing, leading to extremely slow convergence. 
Thus the BSNN has shown initial promise of generating fast-converging starting positions and should continue to be investigated.

# Use
To run the project as is, follow these steps:
1. From the base directory, run "bash lsf.sh". Or "cd src/" followed by "python3 run_simulation.py"
2. Then run "sbatch out/lsf/simulation__startup/simulation__startup.lsf.slurm"
3. Assuming no issues, that is all the steps neccessary to carry out the project. The genetic algorithm will run indefinitely until FOM > .99 is achieved or you have reached 200 generations (to change these terminating conditions, change genetic_alg.py line 252 and/or line 297).
4. You will be able to track the FOM over time within out/results/FOM_history.txt

If you would like to change the genetic algorithm logic, lines 241 - 300 are where the main "thinking" is done. The project should continue as normal to any changes made here. 

If you want to increase the generation size, alter line 515. This will change the number of child configurations generated for each sim in the top 5. 

If you want to alter this project to run a different type of splitter ratio(perhaps 20/80 instead of 33/66), the only change to be made is within sbatch.lsf lines 53 and 54. This is where we define our "ideal" ratio so by altering this you will change how the FOM is calculated and thus what configuration the algorithm will converge to. 

For a more detailed, step-by-step use of this framework see the "expanded use" section.

# Debugging
The number 1 issue you will run into with this implementation is simulation failure. Since we are running the simulations on CRC's preempt partition, they are prone to failure and will usually only work after 2-3 tries. To get around this issue, the base slurm file(lsf.slurm, lines 37-81) will run the sims up to 5 times to ensure that all files are run correctly. Feel free to adjust this as neccessary. You will know this error has occured via the output log which will read "Unsuccessful run of sims".

The next most frequent error is during the startup procedure. Occassionally, run_simulation.py does not always generate the 50 starting sims and as a result not all the neccessary supplementary files are created to carry out the sim. If this happens, just delete the simulation output directory and retry. 

This isn't a debugging issue but you will find that your workspace will quickly become cluttered with output logs from these sims. I recommend running the file "clear_logs.py" from your log file directory occassionally to handle this. 

# Code Base Breakdown
**This repository currently contains lots of irrelevant information for this project so the important details are broken down below**
1. /src/

   a. Genetic_alg.py : Implementation of genetic algorithm. It retrieves the best 5 configurations of the current generation, carries out genetic algorithm logic, and then creates the next generation of sims and supplementary files.
   
   b. Run_simulation.py : This is almost the exact same as genetic_alg.py and could definitely be merged into it. However, currently it is the starting file to generate our first generation of sims based off an initial configuration.
   
   c. Compile_data.py is not neccessary for this project
   
3. /src/functions/

   a. lsf_script.py : Used in genetic_alg and run_simulation to generate an individual .lsf script based off the template we pass into it. This template is defined in the base file(genetic_alg, and run_sim).
   
   b. process_script.py : Adds final formatting and processing to lsf script. Note this is not neccessary and could just be included in the base template. The rest of the file just alters the filename to ensure  unique filenames for each sim.
   
   c. The other files within this directory are not directly relevant to this project.
   
5. /src/lsf_scripts/ **These are probably the most important files in this project.**

   a. lsf.slurm : This is a template of the slurm file needed to run this project on CRC. This file is needed for each generation to carry out running the sims and data storage. It will run sbatch.lsf(see below) and then carry out the execution of the child sims, wait for their completion, run genetic_alg.py, and then submit the next generations slurm file.

   b. sbatch.lsf : This is a "middleman" file that handles interaction between slurm and lumerical. Due to the nature of Lumerical, this file is neccessary to convert our simulation files(.lsf) into .lms files that can then actually be run. It also is responsible for generating individual sub-slurm files for each simulation so that every sim can be run. This file also generates another set of handler files that will load the results of the simulation, calculate the FOM, and store the results so that they can be accessed by our genetic algorithm script. Like lsf.slurm, this is a template file that must be generated for every generation.

   c. All the other files in this directory are not applicable to the project.
7. /out/

   a. Will not contain any useful information until sims are run. However, the simulation files are stored within /out/lsf/{simulation_directory} and the results will be stored in /out/results. The results will include a history of the best FOM, the associated configuration of that FOM, and the associated sim file of that FOM.
   
   b. Also within /out/lsf/ is data_storage. This folder stores every sim config and its associated FOM. It's primary use is for training BSNN.


# Technical Pipeline
This will provide an explanation of the process behind the use:
1. As explained above, run_simulation is called first to generate the first generation of sims as well as a lsf.slurm and sbatch.lsf file.
2. The project is then submitted to CRC via the lsf.slurm file
3. The slurm file then uses Lumerical MODE to carry out the sbatch.lsf script
4. The sbatch script then loops through all of the lsf simulation scripts, creates a corresponding slurm file for the individual sim as well as a data proccessing lsf script.
5. The sbatch script then generates an lms file of our sim. This is neccessary as Lumerical VarFDTD can only recieve lms files.
6. The slurm file for each sim will then run this lms file as well as the data processing script so that the sim is run and data is stored.
7. All of the individual slurm files are grouped together into "run_individual_sims.sh" which will submit each of the individual slurm files to CRC.
8. At this point in the main lsf.slurm file, we call the function "run_on_exit()" which executes run_indivudal_sims.sh to begin simulating all of the individual sims.
9. We then wait for all of the neccessary files to be completed. A sim will be marked as completed when it generates a file ending with ".completed.txt". T
10. This is to indicate that the sim has successfully run and stored all neccessary data. Once the appropriate amount of completed files are genereated, the lsf.slurm file then carries out genetic_alg.py.
11. Genetic_alg.py will create the next generation and its corresponding lsf.slurm file.
12. Then from our original lsf.slurm file, we submit the next generations lsf.slurm file to kickstart the next generation.

# Expanded Use
This section will go over a start to finish process on how to expand this genetic algorithm to run different types of simulations. This will be a general outline, so for a more detailed breakdown on how the code actually works see the following sections: "Genetic Algorithm", "Code Base Breakdown", and "Technical Pipeline". 

1. **Set Up** : Assuming you have just cloned this repo, here are the first couple steps to ensure you have the proper environment
   
   a. First, ensure that you have installed the following python libraries: hashlib, pathlib, sys, numpy, and pandas. This can be done by running "pip install (library)". Additionally, you will need access to CRC/Lumerical to run these simulations. Click here to see how to get access to CRC: https://crc.pitt.edu/requesting-new-account/requesting-new-account/requesting-new-account 

   b. The next step is to get familiar with how the code base currently works as it will be difficult to adapt it for your own project if you don't understand how it works in its current state. You can read the other sections in this repo or you can skim through the code; regardless the most important files to read through are: "run_simulation.py", "genetic_alg.py", "sbatch.lsf" and "lsf.slurm".

   c. It may help to run the project as is first to ensure you are set up properly. See the "Use" section to do so.

2. **Beginning Alterations** : Once you are confident everything is set up correctly, you can begin to make changes in "run_simulation.py"

   a. The first file to take a look at is "run_simulation.py". Despite its name, this file does not actually "run" anything. This is the file that purely generates the first generation of sims and supplementary files.

   b. In "run_simulation", you will need to alter the Setup Script(lines 31 - 226). Setup script is a template for the lumerical simulation you are running. In its current state, it runs a VarFDTD simulation for our 20x20 configuration. Since this is purely a template, some aspects of Setup Script should be subject to change for different sims. For example, since this project runs different kinds of configurations, you will see on line 138 that "hole_array = {configuration}" where {configuration} is a placeholder that gets replaced. Thus for your project, fill in placeholders for whatever aspects you want to change. Everything else will remain constant for every sim.

   c. Next you can get rid of the format_matrix_string function as it is really only specific to this project.

   d. The next step will be highly variant depending on your simulation. Currently, line 324 defines a starting position for our variable element(in this case, a 20x20 configuration). Thus you will have to choose some intial condition or setup for the aspect/s you wish to mutate.

   e. Again, this step will also be highly dependent on your sim. At the moment, the project will make some change(mutation) to the intial condition and then will replace the placeholders in Setup Script to hold this value. We then create an lsf file based off this now "filled in" template. This process is repeated 50 times so that 50 simulation files are created. For your project, how you alter the setup script is up to you, but it is crucial for you to replace the placeholders and create the script.

   f. The rest of the script is focused on file handling and creation of supplementary files needed to run the sim. For now, don't make any more alterations.

   g. Now run "bash lsf.sh" to see if your files are appropriately genereated. They will be stored in /out/lsf/simulation_startup.

4. **Slurm and Sbatch Modifications** : Now that you have your simulation scripts generated, we will alter the processing framework.

   a. The interaction of our simulations and our compute takes place within our slurm scripts. These scripts essentially just tell the cluster what actions to perform, so for us we want to tell it to execute a number of simulations and handle any processing that comes with it. The slurm file we use( found in src/lsf_scripts/lsf.slurm ) handles the workflow for one complete iteration. It executes all the sims in the iteration, ensures no errors occur, carries out the genetic algorithm, and then populates the next generation.

   b. To cater the lsf.slurm script for your sim perform the following actions:

     i. On line 177, change 'mode-solutions-batch' to fit the lumerical solver you are using. This solver should be capable of reading and running .lsf scripts not just sim scripts(.fsp,.lms,etc). Change this again on line 182.

     ii. On line 68, the slurm script waits a certain amount of time before checking if all your sims have been completed. If this time has passed and not all the sims are done or they failed, the slurm script will rerun the sims. Change this time to an approrpriate amount for your sim. For example, a sim that takes 30 seconds to run should have a wait period between 1-2 minutes.

     iii. On lines 98 and 113, change the path appropriately.

   c. The next file to look at is the sbatch.lsf script. This script is used to convert out simulation scripts(.lsf) into actual executable sims(.fsp/.lms/...) as well as generate all the slurm and results proccessing files.

   d. Make the following changes to this script:

     i. Line 34 is used to create a name for the executable simulation. You will need to change this so that the file extension matches whatever solver you are using(.fsp for FDTD, .lms for MODE, etc). You will probably also want to change the variable name to match this.

     ii. Lines 40 - 59 are used to create the results proccessing script. This is another lsf script that will read in the results of the simulation and calculate some metrics. Lines 41 and 42 should remain the same but as they just read in the results, however lines 43 - 55 should be altered to fit your requirements. Currently they get the transmissions through several monitors and calculate a metric based on these parameter. Lines 55 - 59 are neccessary as they store the results of the metrics so we can compare configuarations later. However, the manner on what you store and how you store them is up to you, as long as you can point to a specific, comparable metric then the algorithm will work with minor changes. 

     ii. Lines 63 - 159 create a slurm file for the individual sim currently being inspected so we can run the sims in parallel. For example, if we have 50 sims, then we create 50 slurm files so that they can all be run at once. The only thing that needs to be looked at are lines 117 and 118. Similar to the base slurm file, just change the type of solver to run the sims.

  e. Once these modifications are made, I would recommend doing trial runs of 1 or 2 iterations to ensure that all the sims are running and the correct number of files are being created. 

6. **Genetic Algorithm Modifications** : Now that you've confirmed everything is working, you can modify the genetic algorithm

   a. There are no specific modifications to make to this script as it is almost entirely simulation depedendent, however I will outline the general pieces that are up for modification.

     i. The functions 'read_in_txt_matrix(), read_in_txt_fom(), get_last_fom_values(), write_fom_to_history(), format_matrix_string()" are all data I/O functions. They are specific to the type of simulation we are running but in general they read in the simulation configuration and its associated metrics or they store the metric to history.

     ii. The functions "long_term_LMR(), calc_beta()" are functions used to calculate the mutation strength for child configurations. long_term_LMR sets a base lowest mutation rate based on the previous iterations metric and calc_beta then determines how much the simulations within the generation should be mutated compared to each other. This is in no way the only method to calculate mutation rate and you should change this as neccessary.

     iii. After the function definitions, the general workflow of utilizing these functions to read in data and calculate mutation rate shouldn't change too much but if you would like to change the cap on how many iterations to run then alter line 272.

     iv. The rest of the script is a pretty much a repeat of run_simulation.py so you can reincorporate the changes from that script. 

     
   
