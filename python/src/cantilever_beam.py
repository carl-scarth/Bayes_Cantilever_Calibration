# Returns the deflection of isotropic cantilever beam with rectangular cross-section 
# with breadth b and depth d, length L and Young's modulus E, subject to point load P
# at the tip. Returns deflection at coordinate(s) x, noting the x may be a list. 
# x is bounded in the interval [0,L]. All other inputs are of type float.

def cantilever_beam(x,E,b,d,P,L):
    if type(x) is list:
        delta = [-(P*x_i**2)/(6*E*((b*d**3)/12))*(3*L-x_i) for x_i in x]
    else:
        delta = -(P*x**2)/(6*E*((b*d**3)/12))*(3*L-x)

    return(delta)