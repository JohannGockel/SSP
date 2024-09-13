import numpy as np
from muFFT import FFT
import muGrid as muG
import matplotlib.pyplot as plt



class SpecOps:
    """SpecOps (Spectral Operations) is a class, that contains functions and calculations
    that are often needed, when using pseudo-spectral Methods. The class's purpose is to simplify
    calculations in Jupyter-Notebooks or Python-files.  
    
    Parameters
    ------------
    ng_gp   : tuple
            The number of grid points in each spatial diraction (x,y,z)
    lxyz    :  tuple
            The physical length of each spatial direction (x,y,z)
    nu      : float
            The viscosity of the fluid that will be simulated
    """


    def  __init__(self,nb_gp,lxyz,nu):
        self.nu = nu
        self.nb_gp = nb_gp
        self.lxyz = lxyz
        self.gs = np.array([self.lxyz[0]/self.nb_gp[0],
                            self.lxyz[1]/self.nb_gp[1],
                            self.lxyz[2]/self.nb_gp[2]])
        
        self.fft = FFT(self.nb_gp, engine='pocketfft')

        
        #get real-space coordinates
        self.cords = self.fft.coords
        self.cords[0] *= self.lxyz[0]
        self.cords[1] *= self.lxyz[1]
        self.cords[2] *= self.lxyz[2]

        #get normalised wavenumbers
        self.q = 2*np.pi*self.fft.fftfreq
        self.q[0] /= self.gs[0]
        self.q[1] /= self.gs[1]
        self.q[2] /= self.gs[2]

        self.qx,self.qy,self.qz = self.q
     
    def curl(self,u):
        """Computes the curl of a vector field in real space for a 
            given field 3-dim. field u.
            
            Parameters
            -------------
            u   :   array_like
                    The field that will be curled. Can be complex or real valued.

            Returns
            ------------
            c   :   np.array
                    The curl of u."""

        #initiate u_fields
        u_real = self.fft.real_space_field('u_real',(3,))
        u_hat = self.fft.fourier_space_field('u_hat',(3,))
        u_real.p = u


        #perform fourier-transfrom of u_real
        self.fft.fft(u_real,u_hat)

        #ux,uy,uz = u_hat.p

        #initiate c fields
        c_hat = self.fft.fourier_space_field('c_hat',(3,))
        c = self.fft.real_space_field('c',(3,))
        
        #calculate c_hat field
        c_hat.p = np.cross(self.q,u_hat.p,axisa=0,axisb=0,axisc=0)

        #normalisation of the vector field
        c_hat.p *= 1j*self.fft.normalisation

        #backtransform c_hat to c
        self.fft.ifft(c_hat,c)


        return c.p
    
    def Runge_Kutta_4(self,f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Implements the fourth-order Runge-Kutta method for numerical integration
        of multidimensional fields.

        Parameters
        ----------
        props   :   class object
                    The object, that contains the information of the SpecOps class
        f : function
            The function to be integrated. It should take two arguments: time t
            and field y.
        t : float
            The current time.
        y : array_like
            The current value of the field.
        dt : float
            The time step for the integration.

        Returns
        -------
        dy : np.ndarray
            The increment of the field required to obtain the value at t + dt.
        """
        #claculate Runge-Kutta-Coefficients
        k1 = f(self,t, y)
        k2 = f(self,t + dt / 2, y + dt / 2 * k1)
        k3 = f(self,t + dt / 2, y + dt / 2 * k2)
        k4 = f(self, + dt, y + dt * k3)
            
        return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


    
    
    def rhs(self,t:float,u: np.ndarray) -> np.ndarray:

        """ Computes the right-hand side of the NSe for a given field u and time t
        
        Parameters
        ---------------
        t : float
            The current time
        u : array_like
            The current value of the field. Can be fourier-space or real-space
        
        Returns
        ---------------
        u_rhs,u_rhs_hat : np.ndarray
            The right hand side of the Navier-Stokes-eqaution. 
            If the field u is given in fourier-space, the output 
            is also in fourier-space. Note that in this case, 
            one has to take care of normalization when transforming 
            back to real space.

        Notes
        ----------
        The function includes De-Aliasing of the non-linear Term.
        The function automatically differs if it is given a fourier-
        or a real-space field. It assumes that a fourier-space field
        is always complex valued, while the real-space field is always
        real values.

        """


        #set up the final right-hand-sinde field objects
        uges = self.fft.real_space_field('uges',(3,))
        uges_hat = self.fft.fourier_space_field('uges_hat',(3,))

        
        
        #for transforming u to Fourier-space if needed
        urf = self.fft.real_space_field("u_real_field",(3,))
        u_hat = self.fft.fourier_space_field("u_hat",(3,))

        

        #Check if the given field u is in fourier- or real-space representation
        if u.dtype == 'complex':
            u_hat.p = u
            self.fft.ifft(u_hat,urf)
        
        else: 
            urf.p = u
            self.fft.fft(urf,u_hat)


        #compute non linear term in real space and transform to Fourier-space
        uxomega = self.fft.real_space_field('uxomega',(3,))
        uxomega_hat = self.fft.fourier_space_field('uxomega_hat',(3,))

        #w = 1/2 nabla x u
        omega = 1/2*self.curl(urf.p)
    
        
        #compute vector-product term 2u x w
        uxomega.p = 2*np.cross(urf.p,omega,axisa=0,axisb=0,axisc=0)

        #implement Dealiasing, following the 2/3-Rule introduced by Orzag
        ind = np.where(abs(self.q) < 2*self.q.max()/3)
        uxomega.p[ind] = 0
    
    

        #Transform non linear term
        self.fft.fft(uxomega,uxomega_hat) 

    
        
        #compute diffusing term in Fourier-space directly
        diffu_hat = -self.nu*(self.qx**2+self.qy**2+self.qz**2)*u_hat.p
    
        #compute Pressurefield in Fourier-space
        P_hat = -2j/(self.qx**2+self.qy**2+self.qz**2)*(self.qx*uxomega_hat.p[0]+self.qy*uxomega_hat.p[1]+self.qz*uxomega_hat.p[2])
        P_hat[0,0,0] = 0 #Fix singularity due to qx=qy=qz=0 

        #compute Pressure Gradient in Fourier-space
        Pgrad_hat = np.zeros((3,*P_hat.shape),dtype = 'complex')
        Pgrad_hat[0] = 1j*P_hat*self.qx
        Pgrad_hat[1] = 1j*P_hat*self.qy
        Pgrad_hat[2] = 1j*P_hat*self.qz

        #sum up all calculated term to get the  numerical right hand side of the equation and normalize
        uges_hat.p = (uxomega_hat.p+diffu_hat-Pgrad_hat)*self.fft.normalisation

        #Backtransform rhs to get the real field for time propagation in real space
        self.fft.ifft(uges_hat,uges) 
        if u.dtype == 'complex':
            return uges_hat.p
        else: 
            return uges.p
        
    def random_field_generator(self,A_0):
        """Generates a random velocity field with an energy spectrum according to Kolmogorov's law.
        
        Parameters
        -----------
        props   :   class object
                    The object, that contains the information of the SpecOps class

        A_0 :   float
                The Amplitude of the random generated field at t = 0
        
        Returns
        ----------

        u,uhat  :   np.arrays
                    The random generated velocity field in real and in fourier space."""

        # Compute wavevectors
        q = (2 * np.pi * self.fft.fftfreq.T / self.gs).T

        zero_wavevector = (q.T == np.zeros(3, dtype=int)).T.all(axis=0) #True where wavevector.T = 0, False else
        q_sq = np.sum(q ** 2, axis=0) #Absolute square of \vec{q}

        # Fourier space velocity field
        random_field = np.zeros((3,) + self.fft.nb_fourier_grid_pts, dtype=complex)
        rng = np.random.default_rng()
        random_field.real = rng.standard_normal(random_field.shape)
        random_field.imag = rng.standard_normal(random_field.shape)

        # Initial velocity field should decay as |q|^(-10/3) for the Kolmogorov spectrum
        fac = np.zeros_like(q_sq)

        #create envelope, following the Kolmogorov law
        fac[np.logical_not(zero_wavevector)] = A_0*q_sq[np.logical_not(zero_wavevector)] ** (-5 / 6)

        #use envelope on the random field
        random_field *= fac

        #calculate the q \otimes q, 3x3 matrix
        qxq = np.zeros((3,*q.shape))
        for i in range(3):
            for j in range(3):
                qxq[i][j] = q[i]*q[j]

        u_hat = self.fft.fourier_space_field('u_hat',(3,))
        u = self.fft.real_space_field('u',(3,))

        #build compatible 3x3 identity matrix
        ident = np.zeros_like(qxq)
        for i in range(3):
            ident[i][i] = np.ones(qxq[0][0].shape)

        #Calculate "Transform"-Matrix
        incomp_matrix = ident - 1/q_sq*qxq

        #Fix division by zero
        incomp_matrix[:,:,0,0,0] = 0

        #calculate the randomly generated, incompressible field in fourier space
        for i in range(3):
            for j in range(3):
                u_hat.p[i] += incomp_matrix[i][j]*random_field[j]

        self.fft.ifft(u_hat,u)



        return u.p,u_hat.p
    