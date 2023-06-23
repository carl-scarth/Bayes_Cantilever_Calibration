import numpy as np

class Prior:
# Define class of prior distribution objects containing type of distribution all important statistical attributes
    def __init__(self,name, distribution, param_1, param_2, weight_LHS):
        # Initialise attributes of prior based upon the type of distribution
        self.name = name
        self.distribution = distribution
        self.weight_LHS = weight_LHS
        if (self.distribution == 'Gaussian'):
            # If Gaussian, the prior is described by a mean and Coefficient of Variation
            self.mu = param_1
            self.COV = param_2
        elif (self.distribution == 'Lognormal'):
            self.mu = param_1
            self.s = param_2
        elif (self.distribution == 'Uniform') or (self.distribution == 'Loguniform'):
            # If Uniform or loguniform, the prior is described by a lower and upper bound
            self.lb = param_1
            self.ub = param_2

    @property
    def sigma(self):
        if self.distribution == 'Gaussian':
            if self.mu != 0:
                return self.mu*self.COV/100.0
            else:
                # If the input has zero mean, then the standard deviation is specified directly rather than via the COV
                return self.COV
        else:
            return None
            # Shouldn't be used, but if necessary could actually specify this
        
    @property
    def min(self):
        if self.distribution == 'Gaussian':
            return self.mu - 3.0*self.sigma
        elif self.distribution == 'Lognormal':
            # Defined by +/- 3 standard deviations of log(x)
            return np.exp(self.mu - 3.0*self.s)
        elif (self.distribution == 'Uniform') or (self.distribution == 'Loguniform'):
            return self.lb
        
    @property
    def max(self):
        if self.distribution == 'Gaussian':
            return self.mu + 3.0*self.sigma
        elif self.distribution == 'Lognormal':
            return np.exp(self.mu + 3.0*self.s)
        elif (self.distribution == 'Uniform') or (self.distribution == 'Loguniform'):
            return self.ub