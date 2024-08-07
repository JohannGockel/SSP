import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#for fitting
from scipy.optimize import curve_fit


#just for better plots
import scienceplots

#import Spectral Operations class
from SSP import SpecOps

#specify plot properties
plt.style.use(["science"])
plt.rcParams.update({"figure.figsize": (10,8),
                     "legend.fontsize": 18,
                     "axes.labelsize":20,
                     "figure.labelsize": 20,
                     "xtick.labelsize": 13,
                    "ytick.labelsize": 13})



#setup ticks and tick labels
ticks_label = [r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$",r"$2\pi$"]
ticks = [0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]

#specify viscosity and t=0 amplitude
nu = 1
A_0 = 1

#set up time domain
dt = 0.001
t_beg = 0
t_end = 1.001
t = np.arange(t_beg,t_end,dt)

#set normalization for the color map in the plots
norm = mpl.colors.Normalize(vmin = 0,vmax = A_0)


#initiate periodic domain
nb_grid_points = (50,50,2)
nx,ny,nz = nb_grid_points

physical_sizes = (2*np.pi,2*np.pi,2*np.pi)
lx,ly,lz = physical_sizes

Props = SpecOps(nb_grid_points,physical_sizes,nu) #get Properties and and methods of SSP class

#get normalized wavevector
q = Props.q

#get real field coordinates
x,y,z = Props.cords



'''Analytic solution'''

uana = np.zeros((len(t),3,*x.shape)) #empty array to save the calculatings underneath in

#calculate the velocity field for all times t, see eq. 12 in the protocol 
for i in range(len(t)):
    uana[i,0,:,:,:] = A_0 * np.cos(x)*np.sin(y)*np.exp(-2*nu*t[i])
    uana[i,1,:,:,:] = -A_0 * np.sin(x)*np.cos(y)*np.exp(-2*nu*t[i])
    uana[i,2,:,:,:] = np.zeros_like(x)


'''Set up initial conditions for u'''

u0 = np.zeros((3,*nb_grid_points))
u0[0,:,:,:] = A_0*np.cos(x)*np.sin(y)
u0[1,:,:,:] = -A_0*np.sin(x)*np.cos(y)
u0[2,:,:,:] = np.zeros(nb_grid_points)


'''Time propagation'''

unum = np.zeros((len(t),*u0.shape))#empty array to save the calculatings underneath in
unum[0] = u0 #set initial condition

#Propagte u0 through time
for i in range(len(t)-1):
    unum[i+1] = unum[i] + SpecOps.Runge_Kutta_4(Props,SpecOps.rhs,t[i],unum[i],dt) #Runge Kutta method, calling the methods from SpecOps class
    print(f"Progress: {i/(len(t)-1)*100:.5}%") #Progress indicator
print("Progress: 100%")

#test if shapes of uana and unum are the same
if uana.shape == unum.shape:
    print("We're good")
else:
    print("Something's wrong, I can feel it")



#empty arrays to save the calculatings underneath in
uana_amp = np.zeros((len(t),*unum[0,0,:,:,:].shape))
unum_amp = np.zeros((len(t),*unum[0,0,:,:,:].shape))

#calculate the amplitude fields of the analytic and the numerical field (Amplitude Tensor A eq.13)
for i in range(len(t)):
    uana_amp[i] = np.sqrt(uana[i,0]**2+uana[i,1]**2+uana[i,2]**2)
    unum_amp[i] = np.sqrt(unum[i,0]**2+unum[i,1]**2+unum[i,2]**2)

uana_amp_flat = np.zeros_like(t)
unum_amp_flat = np.zeros_like(t)

#calculate the sum of the flattened amplitude fields for every time step (B-Tensor and sum over l)
for i in range(len(t)):
    unum_amp_flat[i] = sum(unum_amp[i].flatten())
    uana_amp_flat[i] = sum(uana_amp[i].flatten())

#see where the total difference has its maximum --> if not the last time step, the simulation converges
maxdiff = t[np.where(abs(unum_amp_flat-uana_amp_flat) == max(abs(unum_amp_flat-uana_amp_flat)))]
print(maxdiff)

#plotting
plt.figure(figsize=(14,8))
plt.plot(t,uana_amp_flat, label = "Totale Amplitude: Analytische Lösung")
plt.plot(t,unum_amp_flat, label = "Totale Amplitude: Numerische Lösung")
plt.plot(t,abs(unum_amp_flat-uana_amp_flat),label = 'Absolute Differenz')
plt.ylabel(r"Amplitude")
plt.xlabel(r"$t$")
plt.legend()
plt.grid()
plt.show()






#Fitting both progressions in time to the expected time evolution shape (exonential decay)

def exponential_fit(x,nu,A):
    return A*np.exp(-x*nu)

#Get values and errors of the fit parameters 
b_ana,cov_ana = curve_fit(exponential_fit,t,uana_amp_flat)

b_num,cov_num = curve_fit(exponential_fit,t,unum_amp_flat)

#Display values of the fit parameters
print(r"Parameter|",r"Analytical",r"Numerical",r"Absolute Difference")
print('----------------------------------------------------------------')
print(r"nu|q|^2  |",b_ana[0],b_num[0],f'{abs(b_ana[0]-b_num[0])/b_ana[0]*100:.5}%')
print('----------------------------------------------------------------')
print(r"Amplitude|",b_ana[1],b_num[1],f'{abs(b_ana[1]-b_num[1])/b_ana[1]*100:.5}%')




