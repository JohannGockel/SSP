import numpy as np
import matplotlib.pyplot as plt

#just for better plots
import scienceplots
import matplotlib as mpl
from SSP import SpecOps


#initialize Domain
nb_gp = (100,100,2)
l = (2*np.pi,2*np.pi,2*np.pi)


#set inital velocity amplitude and viscosity
A_0 = 0.1
nu = 0.1


#Get properties of the domain and fourier-space-operations
props = SpecOps(nb_gp,l,nu)


#generate Random velocity field with |q|^{-5/6} energy spectrum
u,u_hat = SpecOps.random_field_generator(props,A_0)

#Save the arrays to a file for further use
np.save("random_field_u.npy",u)
np.save("random_field_uhat.npy",u_hat)
