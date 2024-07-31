# A library with useful cosmology routines

import numpy as np
            
def get_cosmo(sim_idx, snap, cosmo_file='cosmology_Aletheia.dat', use_sigma8=True):
    from utilities import make_cosmo_suite   
    cosmo_dict = make_cosmo_suite(f"/ptmp/mesposito/AletheiaSims/aletheia/{cosmo_file}", use_sigma8=use_sigma8)
    cosmo = Cosmology(**cosmo_dict['Aletheia_{}_{}'.format(sim_idx, snap)])
    return cosmo


class Cosmology():

    def __init__(self, omega_cdm = 0.268584094, omega_de = 0.681415906, omega_baryon = 0.05, 
                 hubble = 0.67, ns = 0.96, A_s = None, sigma8 = 0.82755, ReNormalizeInputSpectrum = True, 
                 expfactor = 0.33342518307993363, tau = 0.0952, w0 = -1, wa = 0, Om_EdE = False, de_model = 'lcdm'):
        
        ''' Cosmology class. It contains the cosmological parameters and some useful functions.
        
        Parameters
        -----------
        omega_cdm : float, optional, default = 0.268584094
            Cold dark matter density parameter. Non-physical i.e. this is \Omega_{cdm}
        omega_de : float, optional, default = 0.681415906
            Dark energy density parameter. Non-physical i.e. this is \Omega_{de}
        omega_baryon : float, optional, default = 0.05
            Baryon density parameter. Non-physical i.e. this is \Omega_{b}
        hubble : float, optional, default = 0.67
            Hubble parameter in units of 100 km/s/Mpc i.e. this is h
        ns : float, optional, default = 0.96
            Scalar spectral index of initial power spectrum
        A_s : float, optional, default = None
            Amplitude of initial power spectrum. If None, it is computed from sigma8 (when needed, TODO)
        sigma8 : float, optional, default = 0.82755
            Normalization of the power spectrum at z=0. Used for compatibility with other codes (TODO: use sigma12)
        ReNormalizeInputSpectrum : bool, optional, default = True
            If True, the input power spectrum is renormalized to sigma8 (TODO, just set for retro-compatibility)
        expfactor : float, optional, default = 0.33342518307993363
            Scale factor for this specific cosmology.
        tau : float, optional, default = 0.0952
            Optical depth for reionization. (TODO, just set for retro-compatibility)
        w0 : float, optional, default = -1
            Dark energy equation of state parameter, constant part.
        wa : float, optional, default = 0
            Dark energy equation of state parameter, time-dependent part (linear parametrization).
        Om_EdE : float, optional, default = False
            EdE dark energy model parameter. If False, the dark energy model is LCDM.
        de_model : str, optional, default = 'lcdm'
            Dark energy model. Can be 'lcdm', 'w0wa' or 'EdE'.

        Methods
        -----------
        # Cosmology methods #
        E_z : compute the normalized Hubble parameter at a given scale factor.
        DE_density : compute the dark energy density at a given scale factor.
        w_z : compute the dark energy equation of state parameter at a given scale factor.
        w_eff : compute the effective dark energy equation of state parameter at a given scale factor.

        # Growth methods #
        compute_growth : compute the growth factor and the growth rate at a given scale factor.
        growth_factor : compute the growth factor at a given scale factor.
        growth_rate : compute the growth rate at a given scale factor.

        # Power spectrum methods #
        camb_cleanup : cleans up the camb power spectrum and camb results, so that they are recomputed next time they are needed.
        get_sigmaR : getter function for the camb sigmaR, i.e. the variance of the linear density field smoothed on a scale R.
        get_knl : getter function for knl, i.e. the scale at which the variance of the linear power spectrum is equal to 1.
        get_camb_Pk : getter function for the camb power spectrum. Returns the power spectrum at the desired redshift or scale factor,
            by rescaling the z = 0 camb power spectrum with the growth factor.
        _compute_camb_Pk : computes the camb power spectrum. This is an internal function, for the getter function see get_camb_Pk.

        '''
        
        self.pars = {}
        self.pars['Omega_cdm'] = omega_cdm
        self.pars['Omega_b'] = omega_baryon
        self.pars['Omega_m'] = omega_cdm + omega_baryon
        self.pars['Omega_DE'] = omega_de
        self.pars['Om_EdE'] = Om_EdE
        self.pars['w0'] = w0
        self.pars['wa'] = wa
        self.pars['de_model'] = de_model
        self.pars['sigma8'] = sigma8
        self.pars['h'] = hubble
        self.pars['ns'] = ns
        self.pars['As'] = A_s
        self.pars['tau'] = tau
        self.expfactor = expfactor
        # Setting the physical density parameters
        self.pars['omega_cdm'] = self.pars['Omega_cdm'] * self.pars['h']**2
        self.pars['omega_b'] = self.pars['Omega_b'] * self.pars['h']**2
        self.pars['omega_m'] = self.pars['Omega_m'] * self.pars['h']**2
        self.pars['omega_de'] = self.pars['Omega_DE'] * self.pars['h']**2
        # Setting cached values for the camb power spectrum (maybe this should be done in a different way)
        self.camb_Pk_z0 = None
        self.camb_results_z0 = None
        self.knl = None

    ####### Cosmology methods #######

    def E_z(self, a, deriv=False):
        ''' Compute the normalized Hubble parameter at a given scale factor.
            If deriv=True, it returns the derivative of E_z with respect to a.'''
        if deriv:
            return (1/2)*(-3*self.pars['Omega_m']/a**4 - 2*(1-self.pars["Omega_m"]-self.pars["Omega_DE"])/a**3 + self.DE_density(a, deriv=True))/self.E_z(a)
        else:
            return np.sqrt(self.pars["Omega_m"]/a**3 + (1-self.pars["Omega_m"]-self.pars["Omega_DE"])/a**2 + self.DE_density(a))
            
    def DE_density(self, a, deriv=False):
        ''' Compute the dark energy density at a given scale factor.
            If deriv=True, it returns the derivative of the dark energy density with respect to a.'''
        if self.pars['de_model'] == 'lcdm':
            if deriv:
                return 0
            else:
                return self.pars['Omega_DE']
        else:
            if deriv:
                return self.pars['Omega_DE'] * (-3) * a**(-3*(1+self.w_eff(a))-1) * (a*np.log(a)*self.w_eff(a, deriv=True) + 1+self.w_eff(a))
            else:
                return self.pars['Omega_DE']*a**(-3*(1+self.w_eff(a)))
            
    def w_z(self, a, deriv=False):
        ''' Compute the dark energy equation of state parameter at a given scale factor.
            If deriv=True, it returns the derivative of w_z with respect to a.'''
        
        if self.pars['Om_EdE']:
            b = - 3*self.pars['w0']/( np.log(1/self.pars['Om_EdE']-1) + np.log(1/self.pars['Omega_m']-1) )
        if deriv:
            if self.pars['Om_EdE']:
                return -2*self.pars['w0']*(-b/a)/(1-b*np.log(a))**3
            else:
                return -self.pars['wa']
        else:
            if self.pars['Om_EdE']:
                return self.pars['w0']/(1-b*np.log(a))**2
            else:
                return self.pars['w0']+self.pars['wa']*(1-a)

    def w_eff(self, a, deriv=False):
        ''' Compute the effective dark energy equation of state parameter at a given scale factor.
            If deriv=True, it returns the derivative of w_eff with respect to a.
            In models with varying dark energy equation of state, this is defined as
                w_eff: Omega_DE(a) = Omega_DE * a**(-3*(1+w_eff(a)))'''
        
        # First we calculate the w0 contribution
        if self.pars['Om_EdE']:
            b = - 3*self.pars['w0']/( np.log(1/self.pars['Om_EdE']-1) + np.log(1/self.pars['Omega_m']-1) )
            w0 = self.pars['w0']/(1 - b*np.log(a)) if not deriv else self.pars['w0']*b/(a*(1-b*np.log(a))**2)
        else:
            w0 = self.pars['w0'] if not deriv else 0
        # Then the wa one
        if self.pars['wa'] == 0:
            wa = 0
        else:
            if not deriv:
                wa = np.where(a == 1, 0, self.pars['wa'] * (1+(1-a)/np.log(a)))
            else:
                wa = np.where(a == 1, -self.pars['wa']/2, self.pars['wa'] * (a-a*np.log(a)-1) / (a * np.log(a)**2))
        return w0 + wa
    
    ####### Growth methods #######
    
    def compute_growth(self, a, a0=1e-3, f0=1, solver='odeint'):
        ''' Compute the growth factor and the growth rate at a given scale factor, through solving the generic ODE.
            Works for w0, wa and EdE dark energy models, but not for a generic w(z).
            
            a : float or array
                Scale factor at which to compute the growth factor and the growth rate.
            a0 : float, optional, default = 1e-3
                Initial scale factor value for solving the ODE.
            f0 : float, optional, default = 1
                Initial growth rate value for solving the ODE. The initial D value is set to D0 = a0, assuming matter domination.
            solver : str, optional, default = 'odeint'
                Solver to use for solving the ODE. Can be 'odeint' or 'solve_ivp'.

            return : tuple
                Growth factor and growth rate at the given scale factor.

            '''
        
        from scipy.integrate import solve_ivp, odeint 

        def dy_dt(t, y):
            return [y[1], -(3/t+self.E_z(t, deriv=True)/(self.E_z(t)))*y[1]+y[0]*3*self.pars['Omega_m']/(2*t**5*(self.E_z(t))**2)]
        
        a = np.atleast_1d(a)
        y0 = [a0, f0]      #Initial conditions for D, dD/da
        tspan = (a0, 2)
        if solver == 'solve_ivp':
            sol = solve_ivp(dy_dt, tspan, y0, t_eval=a)
            D, f = sol.y
        else:
            sol = odeint(dy_dt, y0, tfirst=True, t=np.insert(a, 0, a0))
            D = sol[1:, 0]; f = sol[1:, 1]
        
        return D, f*a/D
    
    def growth_factor(self, a, a0=1e-3, solver='odeint'):
        return self.compute_growth(a, a0=a0, solver=solver)[0]
    
    def growth_rate(self, a, a0=1e-3, solver='odeint'):
        return self.compute_growth(a, a0=a0, solver=solver)[1]
    
    ####### Power spectrum methods #######

    def camb_cleanup(self):
        ''' Cleans up the camb power spectrum and camb results, so that they are recomputed next time they are needed.'''
        self.camb_Pk_z0 = None
        self.camb_results_z0 = None
        self.knl = None
        
    def get_sigmaR(self, R, z=None, a=None, Mpc_units=True, recalc=False):
        ''' Getter function for the camb sigmaR, i.e. the variance of the linear density field smoothed on a scale R.
            This is obtained from the z = 0 camb power spectrum,and rescaled to the desired redshift with the growth factor.
        
            Parameters
            -----------
            R : float or array
                Scale at which to compute the variance.
            z : float, optional, default = None
                Redshift at which to compute the variance. If None, the variance is computed at the present time.
            a : float, optional, default = None
                Scale factor at which to compute the variance. If None, it is calculated as 1/(1+z).
            Mpc_units : bool, optional, default = True
                If True, the scale R is in Mpc, otherwise it is in Mpc/h.
            recalc : bool, optional, default = False
                If True, the camb power spectrum is recomputed, otherwise the cached one is used.
                
            return : float or array
                Variance of the linear density field smoothed on a scale R.'''
        
        if a is None:
            a = 1/(1+np.atleast_1d(z)) if z is not None else self.expfactor

        if (self.camb_results_z0 is None) or recalc:
            # We need to have already computed the camb power spectrum to get sigmaR
            # Calling get_camb_Pk with a = 1, will store the cached camb results in self.camb_results_z0
            _ = self.get_camb_Pk(a=1, recalc=recalc)

        sigmaR = self.camb_results_z0.get_sigmaR(R, hubble_units=not Mpc_units)[0]

        a = np.atleast_1d(a)   #TODO: Make this work also when a > 1
        if len(a) == 1:
            if np.abs(a - 1) < 1e-10:   #Which means that a = 1
                return sigmaR
            D = self.growth_factor(np.append(a, 1))
            return sigmaR*D[0]/D[-1]
        else:
            D = self.growth_factor(np.append(a, 1))
            return sigmaR*D[:-1, np.newaxis]/D[-1]
    
    def get_knl(self, minR=0.1, maxR=100, recalc=False):
        ''' Getter function for the camb knl, i.e. the scale at which the variance of the linear power spectrum is equal to 1.
            It is calculated iteratively interpolating in ranges of R between minR(*/)i*10 and maxR(*/)i*10.
            The obtained value is cached in self.knl. If recalc=True, the cached value is ignored and the knl is recomputed.'''
        
        if (self.knl is not None) and not recalc:
            return self.knl
        
        R = np.linspace(minR, maxR, 10000)
        sigmaR = self.get_sigmaR(R)
        minR_idx = np.argmin(np.abs(sigmaR-1))
        if minR_idx == 0:
            return self.get_knl(minR=minR/10, maxR=maxR/10)
        if minR_idx == len(R)-1:
            return self.get_knl(minR=minR*10, maxR=maxR*10)
        self.knl = 2*np.pi/R[minR_idx]

        return self.knl
    
    def get_camb_Pk(self, z = None, a = None, mink=0.005, maxk=0.4, Mpc_units=True, recalc=False):
        ''' Getter function for the camb power spectrum. Returns the power spectrum at the desired redshift or scale factor,
            by rescaling the z = 0 camb power spectrum with the growth factor.
            For how the z = 0 Pk is computed, see _compute_camb_Pk. '''
        # TODO: build interpolator to get camb_Pk at a given k binning
        if (self.camb_Pk_z0 is None) or recalc:
            Pk, self.camb_results_z0 = self._compute_camb_Pk(z=0, mink=mink, maxk=maxk, Mpc_units=Mpc_units)
            self.camb_Pk_z0 = np.array([Pk[0], Pk[1][0]])    # Camb outputs the Pk as Pk[z][k] so we take the z=0 one
        if a is None:
            a = 1/(1+np.atleast_1d(z)) if z is not None else self.expfactor
        a = np.atleast_1d(a)   #TODO: Make this work also when a > 1
        if len(a) == 1:
            if np.abs(a - 1) < 1e-10:   #Which means that a = 1
                return self.camb_Pk_z0
            D = self.growth_factor(np.append(a, 1))
            return [self.camb_Pk_z0[0], self.camb_Pk_z0[1]*D[0]**2/D[-1]**2]
        else:
            D = self.growth_factor(np.append(a, 1))
            return [self.camb_Pk_z0[0], self.camb_Pk_z0[1]*D[:-1, np.newaxis]**2/D[-1]**2]

    def _compute_camb_Pk(self, z = None, a = None, mink=0.005, maxk=0.4, npoints=100, k_per_logint=10, Mpc_units=True):
        ''' Computes the camb power spectrum. This is an internal function, for the getter function see get_camb_Pk.
        
            Parameters
            -----------
            z : float, optional, default = None
                Redshift at which to compute the power spectrum. If None, it is calculated as 1/a - 1.
            a : float, optional, default = None
                Scale factor at which to compute the power spectrum. If None, the power spectrum is computed at the present time.
            mink : float, optional, default = 0.005
                Minimum k value for the power spectrum computation. In units of 1/Mpc if Mpc_units, otherwise h/Mpc.
            maxk : float, optional, default = 0.4
                Maximum k value for the power spectrum computation. In units of 1/Mpc if Mpc_units, otherwise h/Mpc.
            npoints : int, optional, default = 100
                Number of points for the power spectrum computation.
            k_per_logint : int, optional, default = 10
                Number of points per log interval for the power spectrum computation.
            Mpc_units : bool, optional, default = True
                If True, the power spectrum is in units of Mpc, otherwise it is in Mpc/h.
                
            return : tuple
                list with [k, Pk] and camb results.'''
        
        import camb
        from camb import model
        # Returns camb non linear (k, Pk) in units of Mpc (or not if Mpc_units=False)
        h = self.pars['h']

        # Setting the cosmology parameters for CAMB
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100.0*h, ombh2=self.pars['omega_b'], omch2=self.pars['omega_cdm'], 
                           num_massive_neutrinos=0, mnu=0.0,  #TODO: add massive neutrinos
                           tau=self.pars['tau'])
        As = self.pars['As'] if self.pars['As'] is not None else 1e-9
        pars.InitPower.set_params(As=As, ns=self.pars['ns'])    
        pars.NonLinear = model.NonLinear_none

        if self.pars['w0'] != -1 or self.pars['wa'] != 0:
            pars.set_dark_energy(
                w=self.pars['w0'],
                wa=self.pars['wa'],
                dark_energy_model='ppf')
            
        if z == None:
            z = 1/a - 1 if a != None else 1/self.expfactor - 1
        
        pars.set_matter_power(redshifts=np.atleast_1d(z), kmax=10., k_per_logint=k_per_logint)

        pars.WantCls = False
        pars.WantScalars = False
        pars.Want_CMB = False
        pars.DoLensing = False

        results = camb.get_results(pars)

        if Mpc_units:
            mink = mink/h
            maxk = maxk/h

        kk, z_out, pk = results.get_matter_power_spectrum(minkh = mink, maxkh = maxk, npoints = npoints)
        
        if Mpc_units:
            kk = kk*h
            pk = pk/h**3

        if self.pars['As'] is None:
            # Renormalize to sigma8
            sigma8 = results.get_sigma8_0()
            pk *= self.pars['sigma8']**2 / sigma8**2
            self.pars['As'] = As * self.pars['sigma8']**2 / sigma8**2

        return [kk, pk], results
        