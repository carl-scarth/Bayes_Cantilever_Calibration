beamDeflection <- function(x,E,b,d,P,L){
 
  # Returns the deflection of isotropic cantilever beam with rectangular 
  # cross-section with breadth b and depth d, length L and Young's modulus E, 
  # subject to point load P at tip. Returns deflection at coordinate(s) x, 
  # noting the x may be a vector. x is bounded in the interval [0,L]. 
  
  delta = -P*x^2/(6*E*((b*d^3)/12))*(3*L - x)
  
  return(delta)
  
}
