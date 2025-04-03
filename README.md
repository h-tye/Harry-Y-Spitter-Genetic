# Overview
The goal of this project is to inversely design a waveguide splitter to conform to any ratio at the output. This is primarily focused on a 2/3 output at the through port and a 1/3 output at the cross port. To navigate to an optimal solution, we use a genetic algorithm to navigate the search space of possible configurations and locate optimal solutions. Work consulted: "Arbitrary-Direction, multichannel and ultra-compact power splitters by inverse design method” – Ma et. Al. Simulations of design were done via Ansys Lumerical and the University of Pittsburgh Computer Resource Center. 

# Implementation
The waveguide splitter is designed to be implemented on a 2.4 uM x 2.4 uM silicon crystal. In order to test different geometric configurations of this design, we parse the structure into a 20 x 20 grid(i.e. 400 "pixel" structure). Each pixel can be completely etched and contain only air, or can be filled with Silicon(Si). We injected a source of light with wavelengths between 1.5 uM and 1.6 uM into the configuration and then measured the average output across a bandwidth of 500 nm. A figure-of-merit(FOM) was used to judge how much the configuration differed from the ideal output by the following equation: 
FOM = 1 - (abs(T_thru_avg - T_thru_ideal) + abs(T_cross_avg - T_cross_ideal))

![image](https://github.com/user-attachments/assets/bc733dd6-7166-4dd8-9721-12d38ff252da)


A genetic algorithm based off the FOM was then used to selectively search for the best configuration.

# Genetic Algorithm


