import numpy as np
import matplotlib.pyplot as plt

#just for better plots
import scienceplots
import matplotlib as mpl
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable


from SSP import SpecOps

#specify plot properties
plt.style.use(["science"])
plt.rcParams.update({"figure.figsize": (14,8),
                     "legend.fontsize": 29,
                     "axes.labelsize":35,
                     "figure.labelsize":35,
                     "xtick.labelsize": 30,
                    "ytick.labelsize": 30})

#labels for plotting
ticks_label = [r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$",r"$2\pi$"]
ticks = [0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]



#initialize Domain
nb_gp = (50,50,2)
l = (2*np.pi,2*np.pi,2*np.pi)


#set inital velocity amplitude and viscosity
A_0 = 0.1
nu = 0.1


#Get properties of the domain and fourier-space-operations
props = SpecOps(nb_gp,l,nu)

#get real coordinates
x,y,z = props.cords

#get grid spacing
grid_spacing = props.gs

#get fourier-transform object (a bit unnecessary, but has to be done, when working the SpecOps class)
fft = props.fft

#generate Random velocity field with |q|^{-5/6} energy spectrum
#u,u_hat = SpecOps.random_field_generator(props,A_0)

#or load fields from a file
u = np.load("random_field_u_proj.npy")
u_hat = np.load("random_field_uhat_proj.npy")

#find small wave numbers
q = props.q

ind = list(np.where(abs(q) == 0))

for i in range(4):
    ind[i] = ind[i][:5] #only take 5 of them

#set up time domain
dt = 0.001
t_end = 1.001
t = np.arange(0,t_end,dt)


#prepare emtpy arrays, that will be filled during propagation
unum_driven = np.zeros((len(t),*u.shape))
unum_hat_driven = np.zeros((len(t),*u_hat.shape),dtype = 'complex')
unum_hat_driven[0] = u_hat
unum_driven[0] = u

#propagate the fourier space field and save every time step
for i in range(len(t)-1):
    urf_prop = fft.real_space_field('real_field_propagation',(3,))
    ufs_prop = fft.fourier_space_field('fourier_field_propagation',(3,))

    ufs_prop.p = unum_hat_driven[i] + SpecOps.Runge_Kutta_4(props,SpecOps.rhs,t[i],unum_hat_driven[i],dt) #Runge Kutta

    # if i > 500:
    #     ufs_prop.p[tuple(ind)] = unum_hat_driven[500][tuple(ind)]*fft.normalisation
    
    ufs_prop.p[tuple(ind)] = u_hat[tuple(ind)]*fft.normalisation #reset fourier-coefficients, belonging to the frozen modes and normalise

    unum_hat_driven[i+1] = ufs_prop.p #Saving the calculated field for the next propagation

    ufs_prop.p /= fft.normalisation #neutralize Normalisation done in "rhs"

    fft.ifft(ufs_prop,urf_prop) #calculate real space field for plotting

    unum_driven[i+1] = urf_prop.p*fft.normalisation #safe real space field

    print(f"Driven Propagation Progress: {i/(len(t)-1)*100:.5}%") #Progress indicator
print("Driven Propagation Progress: 100%")


#calculate absolute square field for each time step for plotting
unum_driven_abs = np.zeros((len(t),*nb_gp))
unum_hat_driven_abs = np.zeros((len(t),*u_hat[0].shape),dtype = 'complex')

for i in range(len(t)):
    unum_driven_abs[i] = unum_driven[i][0]**2+unum_driven[i][1]**2+unum_driven[i][2]**2

for i in range(len(t)):
    unum_hat_driven_abs[i] = unum_hat_driven[i][0]**2+unum_hat_driven[i][1]**2+unum_hat_driven[i][2]**2
unum_hat_driven_abs = unum_hat_driven_abs.real

#Save all fields
np.save("unum_driven_abs_proj.npy",unum_driven_abs)
np.save("unum_hat_driven_abs_proj.npy",unum_hat_driven_abs)
np.save("unum_driven_proj.npy", unum_driven)









