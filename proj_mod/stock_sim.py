from dataclasses import dataclass 
import numpy as np 
from typing import Optional, Union
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt 
from scipy.integrate import quad

@dataclass
class Vol_params: 
    kappa: float #update speed 
    theta: float #Long run mean 
    sigma: float #std 
    rho: float #covariance of Brownians 
    v0: Union[np.ndarray,float] #starting value(s) 
    
@dataclass
class Jump_params: 
    lam_J: float #frequency 
    mu_J: float #jump mean 
    sig_J: float #jump std 
    
@dataclass
class Market_params: 
    r: float #risk free interest 
    q: float #dividend 
    
@dataclass
class Bates_params: 
    s0: float #starting stock value 
    vol: Vol_params 
    jump: Jump_params
    market: Market_params
    
class Bates_model_risk_free(): 
    def __init__(self, params: Bates_params):
        self.params_=params
        pass 
        
    def create_Brownian_pair(self, n_paths: int, n_steps: int=252, time_len: float=1, seed: Optional[int]=None, verbose=False): 
        v0=self.params_.vol.v0
        if type(v0)== np.ndarray: 
            if v0.shape[0]!= n_paths or len(v0.shape)!=1: 
                ValueError("The shape if v0 is incorrect. ")
            if np.any(v0<0): 
                ValueError("v0 can NOT be negative. ")
        if type(v0)== float: 
            if v0<0: 
                ValueError("v0 can NOT be negative. ")
        if not hasattr(self, "Dt_"): 
            if verbose: 
                print("time delta is missing. ")
            self.Dt_=time_len/n_steps 
            if verbose: 
                print("time delta is created. ")
        sqrt_Dt=np.sqrt(self.Dt_)
        
        if seed is not None: 
            rng=np.random.default_rng(seed)
            rng2=np.random.default_rng(2*seed)
        else: 
            rng=np.random.default_rng()
            rng2=np.random.default_rng()
        
        self.Z1_=rng.standard_normal(size=(n_paths,n_steps+1))     
        self.Z2_=rng2.standard_normal(size=(n_paths,n_steps+1))
            
        rho=self.params_.vol.rho
        self.DW1_=sqrt_Dt * self.Z1_ 
        self.DW2_=rho*sqrt_Dt*self.Z1_ + np.sqrt(1-rho**2)*sqrt_Dt*self.Z2_
        
        if verbose: 
            print("Brownian Pair created. ")
        return self 

    def create_vol(self, n_paths: int, n_steps: int=252, time_len: float=1, seed: Optional[int]=None, verbose=False): 
        v0=self.params_.vol.v0
        if type(v0)== np.ndarray: 
            if v0.shape[0]!= n_paths or len(v0.shape)!=1: 
                ValueError("The shape if v0 is incorrect. ")
            if np.any(v0<0): 
                ValueError("v0 can NOT be negative. ")
        if type(v0)== float: 
            if v0<0: 
                ValueError("v0 can NOT be negative. ")
        if not hasattr(self, "Dt_"): 
            if verbose: 
                print("time delta is missing. ")
            self.Dt_=time_len/n_steps 
            if verbose: 
                print("time delta is created. ")
        if not hasattr(self, "DW2_"): 
            if verbose: 
                print("W2 delta is missing. ") 
            self.create_Brownian_pair(n_paths=n_paths, n_steps=n_steps, time_len=time_len, seed=seed, verbose=verbose) 
            
        self.vol_=np.zeros(shape=(n_paths, n_steps+1)) 
        
        self.vol_[:, 0]=v0 #Regardless if v0 is an array, this should assign correctly. We have already checked that v0 is non-negative. 
        # self.volplus_=self.vol_.copy()
        
        kappa=self.params_.vol.kappa
        theta=self.params_.vol.theta 
        sigma=self.params_.vol.sigma
        
        for next_step in range(1, n_steps+1): 
            vol_cur=self.vol_[:, next_step-1]
            # volplus_cur=self.volplus_[next_step-1]
            if np.any(vol_cur<0): 
                ValueError("There is negative volatility, something went wrong. ")
            self.vol_[:, next_step]=np.maximum(vol_cur + kappa*(theta - vol_cur)*self.Dt_ + sigma*np.sqrt(vol_cur)*self.DW2_[:, next_step-1], 0.0)
        
        if verbose: 
            print("Volatilities created. ") 
        return self 
    
    def bates_create_paths(self, n_paths: int, n_steps: int=252, time_len: float=1, seed: Optional[int]=None, verbose=False, from_start=False): 
        if from_start: 
            if verbose: 
                print("Creating data from the start. ")
            self.create_vol(n_paths=n_paths, n_steps=n_steps, time_len=time_len, seed=seed, verbose=verbose)
            self.create_jump(n_paths=n_paths, n_steps=n_steps, time_len=time_len, seed=seed, verbose=verbose) 
        else: 
            if not hasattr(self, "vol_"): 
                if verbose: 
                    print("Volatility data is missing. ")
                self.create_vol(n_paths=n_paths, n_steps=n_steps, time_len=time_len, seed=seed, verbose=verbose) 
            if not hasattr(self, "DW1_"): 
                if verbose: 
                    print("W1 delta is missing. ")
                self.create_Brownian_pair(n_paths=n_paths, n_steps=n_steps, time_len=time_len, seed=seed, verbose=verbose) 
            if not hasattr(self, "DJ_"): 
                if verbose: 
                    print("jump data is missing. ")
                self.create_jump(n_paths=n_paths, n_steps=n_steps, time_len=time_len, seed=seed, verbose=verbose) 
            if not hasattr(self, "Dt_"): 
                if verbose: 
                    print("time delta is missing. ")
                self.Dt_=time_len/n_steps 
                if verbose: 
                    print("time delta is created. ")
        
        s0=self.params_.s0 
        x0=np.log(s0) 
        r=self.params_.market.r
        q=self.params_.market.q 
        mu_J=self.params_.jump.mu_J
        lam=self.params_.jump.lam_J
        sig_J=self.params_.jump.sig_J
        kappa_J=np.exp(mu_J + 0.5*(sig_J**2))-1 
        v=self.vol_
        
        self.X_=np.full(shape=(n_paths, n_steps+1), fill_value=x0)
        
        self.DX_=(r-q-lam*kappa_J-0.5*v)*self.Dt_ + np.sqrt(v)*self.DW1_ + self.DJ_ 
        self.X_[:,1:]=self.X_[:,1:]+np.cumsum(a=self.DX_[:,:-1], axis=1) 
        
        self.S_=np.exp(self.X_) 
        
        return self 
    
    def create_jump(self, n_paths: int, n_steps: int=252, time_len: float=1, seed: Optional[int]=None, verbose: bool=False): 
        if seed is not None: 
            rng=np.random.default_rng(3*seed)
        else: 
            rng=np.random.default_rng()
        
        if not hasattr(self, "Dt_"): 
            if verbose: 
                print("time delta is missing. ")
            self.Dt_=time_len/n_steps 
            if verbose: 
                print("time delta is created. ")
        lam_J=self.params_.jump.lam_J
        mu_J=self.params_.jump.mu_J
        sig_J=self.params_.jump.sig_J
        jump_num=np.random.poisson(lam=lam_J* self.Dt_, size=(n_paths, n_steps+1))
        has_jump=jump_num>0
        
        self.DJ_=np.zeros(shape=(n_paths,n_steps+1))
        if np.any(has_jump): 
            jump_means=jump_num[has_jump]*mu_J 
            jump_stds=np.sqrt(jump_num[has_jump])*sig_J 
            self.DJ_[has_jump]=rng.normal(loc=jump_means, scale=jump_stds)
        
        if verbose: 
            print("Jump data created. ")
        return self
    
    def draw_paths(self, figsize: tuple=(12,6), time_len: float=1, n_steps: int=252): 
        plt.figure(figsize=figsize) 
        timeline=np.linspace(start=0, stop=time_len, num=n_steps+1)
        if not hasattr(self, "S_"): 
            ValueError("Stock paths does not exist, please generate before drawing. ") 
        for path in self.S_: 
            plt.plot(timeline,path)
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock value")
        plt.title("Risk free Bates model simulation")
        plt.show()
        
    def bates_est_call_payoff_MC(self, strike_price: Union[np.ndarray, float], time_len: float=1, save_payoff: bool=False): 
        if not hasattr(self, "S_"): 
            ValueError("The Bates stock path(s) do(es) not exist, please create them first. ")
        if not hasattr(self, "call_MC_"): 
            self.call_MC_=dict()
        r=self.params_.market.r 
        S_T=self.S_[:,-1]
        if type(strike_price) == np.ndarray: 
            if len(strike_price.shape)>1: 
                ValueError("The algorithm takes only 1d array or float for strike price at this moment. ")
            K=strike_price[...,None]
            key=tuple(strike_price)
        else: 
            K=strike_price
            key=strike_price
        payoff=np.maximum(S_T-K, 0.0) 
        
        disc=np.exp(-r*time_len)
        n_paths=len(S_T)
        
        # print(S_T.shape)
        # print(K.shape)
        # print((S_T-K).shape)
        
        if type(strike_price) == np.ndarray: 
            C0_mu=disc*payoff.mean(axis=1)
            C0_std=disc*(payoff.std(ddof=1, axis=1)/np.sqrt(n_paths))
        else: 
            C0_mu=disc*payoff.mean()
            C0_std=disc*(payoff.std(ddof=1)/np.sqrt(n_paths))
        if save_payoff: 
            self.call_MC_[key]=(C0_mu, C0_std, disc*payoff)
        else: 
            self.call_MC_[key]=(C0_mu, C0_std)
        return self.call_MC_
    
    def bates_CF(self, u, time_len: float=1): 
        ###### There might be issue here 
        
        kappa=self.params_.vol.kappa
        rho=self.params_.vol.rho
        sigma=self.params_.vol.sigma
        theta=self.params_.vol.theta
        v0=self.params_.vol.v0
        s0=self.params_.s0
        r=self.params_.market.r
        q=self.params_.market.q 
        lam_J=self.params_.jump.lam_J
        mu_J=self.params_.jump.mu_J
        sig_J=self.params_.jump.sig_J
        kappa_J=np.exp(mu_J + 0.5*(sig_J**2))-1 
        
        u_arr=np.asarray(u, dtype=np.complex128)
        i=1j
        alpha=-u_arr**2-i*u_arr
        beta=kappa-rho*i*sigma*u_arr
        # gamma=(sigma**2)/2
        d_eu=np.sqrt(beta**2 - alpha*sigma**2)
        g_eu=(beta - d_eu)/(beta + d_eu)
        
        exp_mdT=np.exp(-d_eu*time_len)
        one_m_gexp_mdt = 1-g_eu*exp_mdT
        one_m_g = 1-g_eu
        
        C_eu=((kappa*theta)/(sigma**2))*((beta - d_eu)*time_len - 2*np.log((one_m_gexp_mdt)/(one_m_g)))
        D_eu=((beta - d_eu)*(1-exp_mdT))/(sigma**2 * (one_m_gexp_mdt))
        
        phi=np.exp(i*u_arr*(np.log(s0) + (r-q-lam_J*kappa_J)*time_len) + C_eu + D_eu*v0 + lam_J*time_len*(np.exp(i*u_arr*mu_J - (u_arr**2 * sig_J**2)/2)-1))
        
        return phi 
    
    def bates_est_call_payoff_CF(self, strike_price: Union[np.ndarray,float], u_max: float = 200, du: float = 0.01, time_len: float = 1): 
        ###### There might be issue here 
        
        u=np.arange(start=du, stop=u_max+du, step=du)
        
        # u_arr=u[...,None] #Make it a column vector, so that strike price is the other dimension. 
        
        i=1j
        
        s0=self.params_.s0
        q=self.params_.market.q 
        r=self.params_.market.r 
        
        K=np.asarray(strike_price, dtype=float).ravel()
        k=np.log(K)
        
        phi_eu = self.bates_CF(u=u, time_len=time_len)
        phi_eumi=self.bates_CF(u=u-i, time_len=time_len)
        phi_mi=self.bates_CF(u=-i, time_len=time_len)
        phi_mi = float(np.real_if_close(phi_mi))
        
        phase=np.exp(-i * np.outer(u, k))
        
        # f_1=np.real((np.exp(-i*u_arr*k)*phi_eumi)/(i*u_arr*phi_mi))
        f_1 = np.real(phase * (phi_eumi[...,None]/(i * u[...,None] * phi_mi)))
        I_1 = du * (0.5*f_1[0] + f_1[1:-1].sum(axis=0) + 0.5*f_1[-1])
        # P_1=0.5 + (1/np.pi)*np.trapezoid(y=f_1, x=u, axis=0)
        P_1=0.5 + (1/np.pi)*I_1
        
        # f_2=np.real((np.exp(-i*u_arr*k)*phi_eu)/(i*u_arr))
        f_2 = np.real(phase * (phi_eu[...,None]/(i * u[...,None])))
        I_2 = du * (0.5*f_2[0] + f_2[1:-1].sum(axis=0) + 0.5*f_2[-1])
        # P_2=0.5 + (1/np.pi)*np.trapezoid(y=f_2, x=u, axis=0)
        P_2=0.5 + (1/np.pi)*I_2
        
        #########################################################################################For double checking 
        # f2A = np.real(phase * (phi_eu[:,None] / (1j * u[:,None])))
        # f2B = np.imag(phase * phi_eu[:,None]) / (u[:,None])
        # PA = 0.5 + (1/np.pi) * np.trapezoid(f2A, u, axis=0)
        # PB = 0.5 + (1/np.pi) * np.trapezoid(f2B, u, axis=0)
        # print("max |P2A-P2B| =", np.max(np.abs(PA-PB)))
        
        # kk = float(k[0])

        # def f2_scalar(uu):
        #     return np.real(np.exp(-1j*uu*kk) * self.bates_CF(uu, time_len) / (1j*uu))

        # val2, _ = quad(f2_scalar, du, u_max, limit=200)
        # print("P2_quad vs P2_np:", 0.5 + val2/np.pi, P_2[0])

        # def f1_scalar(uu):
        #     return np.real(np.exp(-1j*uu*kk) * self.bates_CF(uu-1j, time_len) / (1j*uu*phi_mi))

        # val1, _ = quad(f1_scalar, du, u_max, limit=200)
        # print("P1_quad vs P1_np:", 0.5 + val1/np.pi, P_1[0])
        ##########################################################################################
        
        C_0=s0*np.exp(-q*time_len)*P_1 - K*np.exp(-r*time_len)*P_2
        
        if np.isscalar(strike_price): 
            P_1=P_1[0]
            P_2=P_2[0]
            C_0=C_0[0]
            
            key=strike_price
        else: 
            key=tuple(strike_price)
            
        if not hasattr(self, "call_CF_"): 
            self.call_CF_=dict()
            
        self.call_CF_[key]=(P_1,P_2,C_0)
        
        assert abs(self.bates_CF(0.0, time_len) - 1.0) < 1e-12
        assert abs(self.bates_CF(-1j, time_len) - s0*np.exp((r-q)*time_len)) < 1e-8
        
        return self.call_CF_
        