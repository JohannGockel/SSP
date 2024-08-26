from muFFT import FFT
import muGrid as muG
import numpy as np
import matplotlib.pyplot as plt

#just for better plots
from mpl_toolkits.mplot3d import axes3d,art3d
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

dt = 0.001
t_end = 1.001
t = np.arange(0,t_end,dt)


unum_four = np.zeros((len(t),*u.shape))
unum_hat_four = np.zeros((len(t),*u_hat.shape),dtype = 'complex')
unum_hat_four[0] = u_hat
unum_four[0] = u

unum_real = np.zeros((len(t),*u.shape))
unum_real[0] = u




for i in range(len(t)-1):
    unum_hat_four[i+1] = unum_hat_four[i] + SpecOps.Runge_Kutta_4(props,SpecOps.rhs,t[i],unum_hat_four[i],dt)
    print(f"Fourier propagation Progress: {i/(len(t)-1)*100:.5}%") #Progress indicator
print("Progress: 100%")

# for i in range(len(t)-1):
#     unum_real[i+1] = unum_real[i] + SpecOps.Runge_Kutta_4(props,SpecOps.rhs,t[i],unum_real[i],dt)
#     print(f"Real propagation Progress: {i/(len(t)-1)*100:.5}%") #Progress indicator
# print("Progress: 100%")




for i in range(len(t)):
    urf = fft.real_space_field("cache",(3,))
    ufs = fft.fourier_space_field("Cache fourier",(3,))
    ufs.p = unum_hat_four[i]/fft.normalisation
    fft.ifft(ufs,urf)
    unum_four[i] = urf.p*fft.normalisation

unum_abs_four = np.zeros((len(t),*nb_gp))
#unum_abs_real = np.zeros((len(t),*nb_gp))

for i in range(len(t)):
    unum_abs_four[i] = unum_four[i][0]**2+unum_four[i][1]**2+unum_four[i][2]**2

# for i in range(len(t)):
#     unum_abs_real[i] = unum_real[i][0]**2+unum_real[i][1]**2+unum_real[i][2]**2




# abs_tolerance = 1e-14
# test_0 = abs(unum_abs_real - unum_abs_four)

# try:
#     np.testing.assert_allclose(test_0,0,atol = abs_tolerance)
#     print(f'The fourier-space and real-space propagation yield the same results with a tolerance of {abs_tolerance}')
# except:
#     print(f'The fourier-space and real-space propagation yield NOT the same results with a tolerance of {abs_tolerance}')


fig,ax = plt.subplots()
z_show = 0

norm = mpl.colors.Normalize(vmin = unum_abs_four[0][:,:,z_show].min(),vmax = unum_abs_four[0][:,:,z_show].max())
div = make_axes_locatable(ax)
cax = div.append_axes('right','2.5%','2.5%')

ts = 0
p = ax.contourf(x[:,:,z_show].T, y[:,:,z_show].T, unum_abs_four[ts][:,:,z_show],levels = np.linspace(unum_abs_four[ts][:,:,z_show].min(),unum_abs_four[ts][:,:,z_show].max(),15),cmap = plt.cm.turbo,norm = norm)

for i in [0,100,500,1000]:
    ts = i
    ax.clear()
    p = ax.contourf(x[:,:,z_show].T, y[:,:,z_show].T, unum_abs_four[ts][:,:,z_show],levels = np.linspace(unum_abs_four[ts][:,:,z_show].min(),unum_abs_four[ts][:,:,z_show].max(),15),cmap = plt.cm.turbo, norm = norm)
    ax.streamplot(x[:,:,z_show].T,y[:,:,z_show].T,unum_four[ts,0,:,:,z_show],unum_four[ts,1,:,:,z_show],density = 1.5,color = unum_abs_four[ts,:,:,z_show]*2,cmap = "Blues", norm = norm)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_label,fontsize = 30)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks_label,fontsize = 30)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.margins(x=0,y=0)
    cbar = fig.colorbar(p,ax = ax,cax=cax, label = r'$|\vec{u}(\vec{x})|^2$')
    cbar.ax.tick_params(labelsize = 27)
    plt.tight_layout()
    plt.savefig(f'NSE_doc/Bilder/RFTP/random_field_t{ts}_fourier',dpi = 400)


    # for i in [0,100,500,1000]:
    #     fig,ax = plt.subplots(figsize = (10,6))
    #     ts = i
    #     p = ax.contourf(x[:,:,z_show].T, y[:,:,z_show].T, unum_abs_real[ts][:,:,z_show],levels = np.linspace(unum_abs_real[ts][:,:,z_show].min(),unum_abs_real[ts][:,:,z_show].max(),15),cmap = plt.cm.turbo,extend = 'both')
    #     ax.streamplot(x[:,:,z_show].T,y[:,:,z_show].T,unum_real[ts,0,:,:,z_show],unum_real[ts,1,:,:,z_show],density = 1.5,color = unum_abs_real[ts,:,:,z_show]*2,cmap = "Blues")
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels(ticks_label,fontsize = 13)
    #     ax.set_yticks(ticks)
    #     ax.set_yticklabels(ticks_label,fontsize = 13)
    #     ax.set_xlabel(r"$x$")
    #     ax.set_ylabel(r"$y$")
    #     ax.margins(x=0,y=0)
    #     # ax.text(y=1.1, x=1, s=r"$t = {t:.4}$".format(t=t[i]),  transform=ax.transAxes,fontsize = 20)
    #     # ax.text(y=1.04, x=1, s=r"$\nu = {nu:.4}$".format(nu = float(nu)),  transform=ax.transAxes,fontsize = 20)
    #     cbar = fig.colorbar(p, label = r'$|\vec{u}(\vec{x})|^2$')
    #     cbar.ax.tick_params(labelsize = 10)
    #     fig.savefig(f'NSE_doc/Bilder/RFTP/random_field_t{ts}_real', dpi = 400)

