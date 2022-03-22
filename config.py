import numpy as np

poles_nested = ['eds', "edshat", 'spectral','greedy','automatic','fully_automatic', "polynomial"]
poles_not_nested = ['zolo','zolohat','bura']
poles = poles_nested + poles_not_nested

poles_on_spectrum = ["zolo", "eds", "automatic", "fully_automatic"]
poles_on_R_neg = ["zolohat", "edshat", "bura"]


nested = {}
for p in poles_nested:
    nested[p] = True
for p in poles_not_nested:
    nested[p] = False
    
    
# For EDS
EDS_PARAM = 1/np.sqrt(2)


# Newton Parameter for epole computation
newton_tolerance = 1e-12
newton_maxit = 10000

# parameter for computing initial value for Newton in automatic poles
initial_refinement = 20  