import numpy as np
from scipy.stats import norm, uniform, loguniform, lognorm, gamma
from scipy.stats.qmc import LatinHypercube
from skopt.space import Space
from skopt.sampler import Lhs
from Prior import Prior
      
def transformed_LHS(input_list, N_train, sampler_package = "scipy", sampler_kwargs = {}):
    # Takes a list describing input random variables, generates a set of 
    # N_train Latin Hypercube samples using packack "sampler_package", and 
    # scales them onto the correct range, or if required, weights them 
    # according to the prior PDF. Each entry of input_list contains a list
    # of parameters for each random variable, with entries:
    # 0 : Name
    # 1 : Distribution
    # 2 : Parameter 1
    # 3 : Parameter 2
    # 4 : Boolean indicating whether or not to weight according to the RV PDF
    # Where parameters 1 and 2 depend upon the distribution
    
    # Create a list of prior objects containing details of priors for each input
    d = len(input_list) # Number of inputs
    priors = []
    for item in input_list:
        priors.append(Prior(item[0],item[1],item[2],item[3],item[4]))      

    # Generate a set of N samples using the required Latin Hypercube sampler
    if sampler_package == "scipy":
        sampler = LatinHypercube(d=d, optimization = "random-cd")
        FLHS = sampler.random(N_train)
    elif sampler_package == "scikit-optimize":
        sampler = Lhs(**sampler_kwargs)
        space = Space([(0.0, 1.0) for entry in input_list])
        FLHS = sampler.generate(space.dimensions, N_train)
        FLHS = np.array(FLHS)

    # Weight the Latin Hypercube Sample acccording to the prior PDF if required
    # otherwise bound between max and min values (taken as +/- 3 standard 
    # deviations if Gaussian)
    xLHS = np.empty([N_train, d])
    # It would be good to have this as a method of the prior. 
    # It might be possible to do this via the rv_continuous class of scipy
    # without the conditional
    for i, item in enumerate(priors):
        if not item.weight_LHS:
            xLHS[:,i] = [uniform.ppf(xij, loc = item.min, scale = (item.max - item.min)) for xij in FLHS[:,i]]
        else:
            if item.distribution == 'Gaussian':
                xLHS[:,i] = [norm.ppf(xij, loc = item.mu, scale = item.sigma) for xij in FLHS[:,i]]
            elif item.distribution == 'Lognormal':
                # scipy respresents lognormal distribution in an unusual way
                xLHS[:,i] = [lognorm.ppf(xij, item.s, loc = 0.0, scale = np.exp(item.mu)) for xij in FLHS[:,i]]
            elif item.distribution == 'Uniform':
                # Note, the scipy uniform distribution is bounded by [loc, loc + scale]
                xLHS[:,i] = [uniform.ppf(xij, loc = item.lb, scale = (item.ub - item.lb)) for xij in FLHS[:,i]]
            elif item.distribution == 'Loguniform':
                xLHS[:,i] = [loguniform.ppf(xij, item.lb, item.ub) for xij in FLHS[:,i]]
            elif item.distribution == 'Loggamma':
                # Take inverse of gamma pdf then take the exponent (as this is an inverse transformation)
                xLHS[:,i] = [np.exp(gamma.ppf(xij, item.shape, scale = 1.0/(item.rate))) for xij in FLHS[:,i]]
    
    return(xLHS)