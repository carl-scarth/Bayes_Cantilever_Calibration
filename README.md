
# Bayes_Cantilever_Calibration

This repository contains Bayesian uncertainty quantification case studies built around a toy problem, considering the displacement of a cantilever Euler-Bernoulli beam with rectangular cross section and a point load at the tip. These studies were created to test code for fitting Gaussian process emulators and Bayesian model calibration. Code was initially written in R and Stan, then later converted to python (and pymc) for improved efficiency. Both versions of the code are provided. Each of the scripts in the main "R" and "python" subdirectories are self-contained case studies. Contents include:

- cantilever_calibration.py, cantlever_calibration.R: Demonstrative 1D model calibration example using synthetic experimental data constructed from model output + noise.

- cantilever_emulator.py, cantilever_emulator.R: Demonstative example for fitting a Gaussian process emulator across a 2D input space. Useful for testing if the emulator gives reasonable predictions before using for calibration. The python version of the code compares the posterior distributions of the emulator hyperparameters with a Maximum Likelihood Estimate.

- tip_emulator.py: Demonstrative example using Gaussian process emulators in uncertainty propagation of a of a 5D input onto a 1D (univariate) output, taken as the maximum beam displacement. Probabilty Density Functions of the emulated output and model are compared.

---

How to run:

For best results, run the python scripts from the command line via a conda environment as described in the installation procedure for pymc, see:  
<https://www.pymc.io/projects/docs/en/stable/installation.html>

The R scripts may be run from [RStudio](https://posit.co/download/rstudio-desktop/). The <code>setwd()</code> line near the top of each script must first be modified to the sub-directory containing the R scripts on the user's system. Before running RStan you will need to configure your R installation to be able to compile C++ code using the latest version of Rtools. See: 

<https://github.com/stan-dev/rstan/wiki/Rstan-Getting-Started>  
<https://cran.r-project.org/bin/windows/Rtools/>

---

Dependencies:
Python scripts have been implemented and tested using:
Python 3.13.2, pymc 5.22.0, seaborn 0.13.2, numpy 2.2.4, scipy 1.15.2, pandas 2.2.3, arviz 0.21.0, pytensor 2.30.2, scikit-optimize 0.10.2, matplotlib 3.10.1, scikit-learn 1.6.1

R scripts have been implemented and tested using:
R 4.4.3, Rtools 44, StanHeaders 2.32.10, Rstan 2.32.7, data.table 1.17.0, maximin 1.0-6, plot3D 1.4.1, plot3Drgl 1.0.4, colormap 0.1.4
