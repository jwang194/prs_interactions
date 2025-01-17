You will need a working conda installation in order to use these files.

Before doing anything else, please run startup.sh

This script will prepare a conda environment for the scripts in this directory. It will then activate the environment, produce an appropriate iPython kernel, and open Jupyter. 

From there, you may open interaction_regression.ipynb and begin testing code. All data should be formatted as shown in 
example_prs.txt
example_betas.txt

If you would like to use new simulated data, the code used to generate the example files is in generate_example.py. The generative model defined there is for a typical multivariate liability model, but with thresholds determined by single liabilities AND pairwise products of liabilities. 
