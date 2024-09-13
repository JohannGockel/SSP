import numpy as np
import matplotlib.pyplot as plt

#just for better plots
import scienceplots
import matplotlib as mpl
from SSP import SpecOps



plt.style.use(["science"])
plt.rcParams.update({"figure.figsize": (14,8),
                     "legend.fontsize": 25,
                     "axes.labelsize":25,
                     "figure.labelsize": 25,
                     "xtick.labelsize": 18,
                    "ytick.labelsize": 18})

#Load the necessary field
unum_hat_driven_abs = np.load("unum_hat_driven_abs.npy")

#specify the necessary properties of the field
nb_gp = (50,50,2)
l = (2*np.pi,2*np.pi,2*np.pi)
nu = 0.1


props = SpecOps(nb_gp,l,nu)

#Get q for plotting
q = props.q

'''Plot the Energy and Dissipation spectra for several time points'''
zshow = 0
yshow = 40
ts = 0
q_E_spec = np.linspace(1,250,1000)
q_E_spec = abs(q[0,:,yshow,zshow])

tsm = np.array([[0,100,500],[700,800,1000]])
dt = 0.001
t_end = 1.001
t = np.arange(0,t_end,dt)

colors = np.array([["red","green","magenta"],["orange","lime","black"]])

E_norm = np.zeros((*tsm.shape,len(q[0,:,yshow,zshow])))

#Plot the Energy spectrum
fig,ax = plt.subplots()
for i in range(2):
    for j in range(3):
        
        #Calculate the normalised Energy spectrum
        E_norm[i,j] = unum_hat_driven_abs.real[tsm[i,j],:,yshow,zshow]/2
        
        #Plot the Energy spectrum
        ax.plot(np.fft.fftshift(abs(q[0,:,yshow,zshow])), np.fft.fftshift(E_norm[i,j]), marker = 'x',ls = '', \
            label = r"$t = {ts:.2}$".format(ts = t[tsm[i,j]]), ms = 12, color = colors[i,j])
        
#Plot the Kolmogorov law for comparison
ax.plot(q_E_spec,q_E_spec**(-5/3),label = r"$q^{-5/3}$",color = 'blue')
ax.set_ylabel(r"$E(q_x,t)/ \Sigma E(q_x,t)$")
ax.set_xlabel(r"$q_x$")
ax.semilogx()
ax.semilogy()
ax.legend(frameon = True, loc = 'upper right')
ax.grid()
ax.set_xlim(0.9,220)
fig.savefig(f"NSE_doc/Bilder/energy_spectrum_dealias/e_spec_de_tall_over.png")


#Plot the dissipation spectrum
fig,ax = plt.subplots()
for i in range(2):
    for j in range(3):
        
        #Calculate the normalised dissipation specturm
        D_norm = 2*nu*q_E_spec**2*E_norm[i,j]

        ax.plot(np.fft.fftshift(abs(q[0,:,yshow,zshow])), np.fft.fftshift(D_norm), marker = 'x',ls = '', \
                 label = r"$t = {ts:.2}$".format(ts = t[tsm[i,j]]), ms = 12, color = colors[i,j])

#Plot the reference for the dissipation spectrum which is proportional to q^{-1/3}
ax.plot(q_E_spec,nu*q_E_spec**(1/3),label = r'$ \nu q^{1/3}$', color = "blue")
ax.set_ylabel(r"$D(q_x,t)/ \Sigma D(q_x,t)$")
ax.set_xlabel(r"$q_x$")
ax.semilogx()
ax.semilogy()
ax.legend(frameon = True, loc = 'upper right')
ax.grid()
ax.set_xlim(0.9,220)
fig.savefig(f"NSE_doc/Bilder/diss_spectrum_dealias/diss_spec_de_tall_over.png")