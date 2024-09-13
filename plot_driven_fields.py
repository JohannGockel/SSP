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


'''Load random fields'''
unum_driven_abs = np.load("/home/johann_gockel/SSP/unum_driven_abs.npy")
unum_driven = np.load("/home/johann_gockel/SSP/unum_driven.npy")


'''Plot the driven field'''

fig,ax = plt.subplots()
z_show = 0

norm = mpl.colors.Normalize(vmin = unum_driven_abs[0][:,:,z_show].min(),vmax = unum_driven_abs[0][:,:,z_show].max())
div = make_axes_locatable(ax)
cax = div.append_axes('right','2.5%','2.5%')

ts = 0
p = ax.contourf(x[:,:,z_show].T, y[:,:,z_show].T, unum_driven_abs[ts][:,:,z_show],levels = np.linspace(unum_driven_abs[ts][:,:,z_show].min(),unum_driven_abs[ts][:,:,z_show].max(),15),cmap = plt.cm.turbo,norm = norm)

for i in [0,1,100,500,1000]:
    ts = i
    ax.clear()
    p=ax.contourf(x[:,:,z_show].T, y[:,:,z_show].T, unum_driven_abs[ts][:,:,z_show],levels = np.linspace(unum_driven_abs[ts][:,:,z_show].min(),unum_driven_abs[ts][:,:,z_show].max(),15),cmap = plt.cm.turbo,norm = norm)
    ax.streamplot(x[:,:,z_show].T,y[:,:,z_show].T,unum_driven[ts,0,:,:,z_show],unum_driven[ts,1,:,:,z_show],density = 3,color = unum_driven_abs[ts,:,:,z_show]*2,cmap = "Blues",norm = norm)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_label,fontsize = 28)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks_label,fontsize = 28)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.margins(x=0,y=0)
    cbar_p = fig.colorbar(p,ax = ax,cax = cax, label = r'$|\vec{u}(\vec{x})|^2$')
    cbar_p.ax.tick_params(labelsize = 28)
    plt.tight_layout()
    plt.savefig(f"/home/johann_gockel/SSP/NSE_doc/Bilder/mode_driven/turbulence_over_force_{ts}.png")