import numpy as np
import matplotlib.pyplot as plt

#just for better plots
import scienceplots


plt.style.use(["science"])
plt.rcParams.update({"figure.figsize": (14,8),
                     "legend.fontsize": 25,
                     "axes.labelsize":25,
                     "figure.labelsize": 25,
                     "xtick.labelsize": 18,
                    "ytick.labelsize": 18})

#the measured Data
A0 = np.array([11,10,8,6,3,1,0.5,0.1,0.05,0.01,0.005,0.001])
numin = np.array([1.94,1.7,1.5,1,0.5,0.1,0.07,0.04,0.001,4e-5,1e-5,4e-7])
dnumin = np.array([0.01,0.1,0.1,0.1,0.1,0.1,0.01,0.04,0.001,1e-5,1e-5,1e-7])

numax = np.array([2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1,2.1])
dnumax = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

#Plotting
fig,ax = plt.subplots()
(_, caplines, _,) = ax.errorbar(A0,numin,yerr=[np.zeros(len(dnumin)),dnumin],lolims = True, marker = 'x',ls = '--',color = 'red', label = r'$\nu_\mathrm{min}$',ms = 15)
caplines[0].set_marker('_')
caplines[0].set_markersize(20)
(_, caplines, _,) = ax.errorbar(A0,numax,yerr =[dnumax,np.zeros(len(dnumax))],uplims = True, marker = 'x',ls = '--',color = 'blue',label = r'$\nu_\mathrm{max}$',ms = 15)
caplines[0].set_marker('_')
caplines[0].set_markersize(20)
ax.fill_between(A0,numin,numax, alpha = 0.2, color = 'orange')
ax.fill_between(A0,numin+dnumin,numax-dnumax, alpha = 0.2, color = 'green')
ax.text(x = 3e-3,y = 7e-3,s ='Stabilitätsgebiet', fontsize = 30)
ax.set_ylabel(r'$\nu$')
ax.set_xlabel(r'$A_0$')
plt.semilogx()
plt.semilogy()
plt.legend()
plt.grid()
plt.savefig('stabilitäts_diagramm_0.0005.png',dpi = 500)
plt.show()

