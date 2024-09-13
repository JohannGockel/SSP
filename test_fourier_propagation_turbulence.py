from muFFT import FFT
import muGrid as muG
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
                     "legend.fontsize": 35,
                     "axes.labelsize":35,
                     "figure.labelsize": 35,
                     "xtick.labelsize": 30,
                    "ytick.labelsize": 30})

#labels for plotting
ticks_label = [r"$0$",r"$\frac{\pi}{2}$",r"$\pi$",r"$\frac{3\pi}{2}$",r"$2\pi$"]
ticks = [0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]

#initialize Domain
nb_gp = (50,50,2)
l = (2*np.pi,2*np.pi,2*np.pi)

#set inital velocity amplitude
A_0= 0.1
nu = 0.1

#Get properties of the domain and all fourier-space-operations
props = SpecOps(nb_gp,l,nu)

#get real coordinates
x,y,z = props.cords

grid_spacing = props.gs

#get fourier-transform object
fft = props.fft

u,u_hat = SpecOps.random_field_generator(props,A_0)

#set up time domain
dt = 0.001
t_end = 1.001
t = np.arange(0,t_end,dt)

#set up arrays for the field, that will be propagated in fourier-space
unum_four = np.zeros((len(t),*u.shape))
unum_hat_four = np.zeros((len(t),*u_hat.shape),dtype = 'complex')
unum_hat_four[0] = u_hat
unum_four[0] = u

#set up arrays for the field, that will be propageted in real-space
unum_real = np.zeros((len(t),*u.shape))
unum_real[0] = u



#propagate field in fourier-space
for i in range(len(t)-1):
    unum_hat_four[i+1] = unum_hat_four[i] + SpecOps.Runge_Kutta_4(props,SpecOps.rhs,t[i],unum_hat_four[i],dt)
    print(f"Fourier propagation Progress: {i/(len(t)-1)*100:.5}%") #Progress indicator
print("Progress: 100%")

#propagate field in real-space
for i in range(len(t)-1):
    unum_real[i+1] = unum_real[i] + SpecOps.Runge_Kutta_4(props,SpecOps.rhs,t[i],unum_real[i],dt)
    print(f"Real propagation Progress: {i/(len(t)-1)*100:.5}%") #Progress indicator
print("Progress: 100%")



#back transform fourier-field !!!normalisation!!!
for i in range(len(t)):
    urf = fft.real_space_field("cache",(3,))
    ufs = fft.fourier_space_field("Cache fourier",(3,))
    ufs.p = unum_hat_four[i]/fft.normalisation
    fft.ifft(ufs,urf)
    unum_four[i] = urf.p*fft.normalisation

unum_abs_four = np.zeros((len(t),*nb_gp))
unum_abs_real = np.zeros((len(t),*nb_gp))

#calculate absolute fields
for i in range(len(t)):
    unum_abs_four[i] = unum_four[i][0]**2+unum_four[i][1]**2+unum_four[i][2]**2

for i in range(len(t)):
    unum_abs_real[i] = unum_real[i][0]**2+unum_real[i][1]**2+unum_real[i][2]**2



#check compatibility of both fields
abs_tolerance = 1e-14
test_0 = abs(unum_abs_real - unum_abs_four)

try:
    np.testing.assert_allclose(test_0,0,atol = abs_tolerance)
    print(f'The fourier-space and real-space propagation yield the same results with a tolerance of {abs_tolerance}')
except:
    print(f'The fourier-space and real-space propagation yield NOT the same results with a tolerance of {abs_tolerance}')

