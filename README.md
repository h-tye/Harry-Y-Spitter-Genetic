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
  2. Run sim on configurations, retrieve the 5 best configurations(based by FOM)
  3. Generate 9 children for each of the best 5
     a. Children are made by randomly toggling a certain number of pixels within the parent structure. Hence we are introducing "mutations" to the children.
     b. The best configuration will have the lowest mutation rate(LMR) and we gradually increase the mutation rate for the next 4. For example, if our LMR is 1
     then the best configuration will generate 9 children that have 1 randomly mutated pixel. The second best configuartion will generate 9 children that have 2
     mutated pixels, etc.
  4. We then generate 5 more moderately random sims, typically with a mutation rate of 25 to ensure genetic variance within the generation.
  5. This then comprises the next generation where we will run steps 2-4 until a desirable FOM is achieved or some other terminal condition is reached.
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
BSNN initially was an attempt to train a CNN to recognize optimal features within our configurations and then generate a configuartion of high accuracy(FOM>.99). It was trained of 27K test configurations and acheived low loss within the test data. However, when run in a sim around 20% loss was seen and thus was not an accurate enough model to carry out its function. 
That being said, when a BSNN generated configuration was used as a starting configuration to generate the first generation(as opposed to randomly generating children), convergence to high FOM regions(FOM > .85) was achieved in 2-7 generations. This is opposed to our original method which took 20-30 generations to reach the same level of FOM. Additionally, when a non-BSNN generated configuration with similar FOM was used as a starting position, the FOM degenerated for 10+ generations before increasing, leading to extremely slow convergence. 
Thus the BSNN has shown initial promise of generating fast-converging starting positions and should continue to be investigated.



