import numpy as np
from muFFT import FFT
import muGrid as muG
import matplotlib.pyplot as plt


class SpecOps:

    def  __init__(self,nb_gp,lxyz,nu):
        self.nu = nu
        self.nb_gp = nb_gp
        self.lxyz = lxyz
        self.gs = np.array([self.lxyz[0]/self.nb_gp[0],
                            self.lxyz[1]/self.nb_gp[1],
                            self.lxyz[2]/self.nb_gp[2]])
        
        self.fft = FFT(self.nb_gp, engine='pocketfft')

        

        self.cords = self.fft.coords
        self.cords[0] *= self.lxyz[0]
        self.cords[1] *= self.lxyz[1]
        self.cords[2] *= self.lxyz[2]

        self.q = 2*np.pi*self.fft.fftfreq
        self.q[0] /= self.gs[0]
        self.q[1] /= self.gs[1]
        self.q[2] /= self.gs[2]

        self.qx,self.qy,self.qz = self.q
     
    def curl(self,u):
        """Computes the curl of a vector field in real space for a 
            given field 3-dim. field u"""

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

        #cx,cy,cz = c.p


        return c.p
    
    def mappable_color_amp(u,cm):

        '''Computes Color magnitudes given a 3-dim Vektorfield and a Matplotlib Colormap for 3D-Quiver-Plotting'''
    
        cx,cy,cz = u
        c = np.sqrt(cx**2+cy**2+cz**2)
        # Flatten and normalize
        c = c.ravel()
        # Colormap

        c = cm(c)
        return c
    
    def Runge_Kutta_4(self,f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Implements the fourth-order Runge-Kutta method for numerical integration
        of multidimensional fields.

        Parameters
        ----------
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
            
        k1 = f(self,t, y)[0]
        k2 = f(self,t + dt / 2, y + dt / 2 * k1)[0]
        k3 = f(self,t + dt / 2, y + dt / 2 * k2)[0]
        k4 = f(self, + dt, y + dt * k3)[0]
        
        return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


    
    
    def rhs(self,t:float,u: np.ndarray) -> np.ndarray:

        """ Computes the right-hand side of the NSe for a given field u and time t
        
        Differs if a fourier-space or a real-space field is inserted.
        Assumes, that the fourier-space field is 'complex' and the real-space field is 'float64'.
        """
        
        #set up the final right-hand-sinde field objects
        uges = self.fft.real_space_field('uges',(3,))
        uges_hat = self.fft.fourier_space_field('uges_hat',(3,))

        
        
        #for transforming u to Fourier-space if needed
        urf = self.fft.real_space_field("u_real_field",(3,))
        u_hat = self.fft.fourier_space_field("u_hat",(3,))

        urf.p = u

        #compute non linear term in real space and transform to Fourier-space
        uxomega = self.fft.real_space_field('uxomega',(3,))
        uxomega_hat = self.fft.fourier_space_field('uxomega_hat',(3,))


        omega = 1/2*self.curl(u) # w = 1/2 nabla x u
    
        
        #compute vector-product term 2u x w
        uxomega.p = 2*np.cross(u,omega,axisa=0,axisb=0,axisc=0)
    
    


        self.fft.fft(uxomega,uxomega_hat) #Transform nonlinear term
        
        
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

        self.fft.ifft(uges_hat,uges) #Backtransform rhs to get the real field for time propagation in real space
        
        return uges.p,uges_hat.p