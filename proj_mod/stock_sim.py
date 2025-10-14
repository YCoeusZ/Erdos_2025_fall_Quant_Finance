from dataclasses import dataclass 
import numpy as np 
from typing import Optional, Union
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt 

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
        
        self.Z1_=rng.standard_normal(size=(n_paths,n_steps))     
        self.Z2_=rng2.standard_normal(size=(n_paths,n_steps))
            
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
            
        self.vol_=np.zeros(shape=(n_paths, n_steps)) 
        
        self.vol_[:, 0]=v0 #Regardless if v0 is an array, this should assign correctly. We have already checked that v0 is non-negative. 
        # self.volplus_=self.vol_.copy()
        
        kappa=self.params_.vol.kappa
        theta=self.params_.vol.theta 
        sigma=self.params_.vol.sigma
        
        for next_step in range(1, n_steps): 
            vol_cur=self.vol_[:, next_step-1]
            # volplus_cur=self.volplus_[next_step-1]
            if np.any(vol_cur<0): 
                ValueError("There is negative volatility, something went wrong. ")
            self.vol_[:, next_step]=np.maximum(vol_cur + kappa*(theta - vol_cur)*self.Dt_ + sigma*np.sqrt(vol_cur)*self.DW2_[:, next_step-1], 0.0)
        
        if verbose: 
            print("Volatilities created. ") 
        return self 
    
    def create_paths(self, n_paths: int, n_steps: int=252, time_len: float=1, seed: Optional[int]=None, verbose=False): 
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
        
        self.X_=np.full(shape=(n_paths, n_steps), fill_value=x0)
        
        self.DX_=(r-q-lam*kappa_J-0.5*v)*self.Dt_ + np.sqrt(v)*self.DW1_ + self.DJ_ 
        self.X_[:,1:]=self.X_[:,1:]+np.cumsum(a=self.DX_[:,:-1], axis=1) 
        
        self.S_=np.exp(self.X_) 
        
        return self 
    
    def create_jump(self, n_paths: int, n_steps: int=252, time_len: float=1, seed: Optional[int]=None, verbose=False): 
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
        jump_num=np.random.poisson(lam=lam_J* self.Dt_, size=(n_paths, n_steps))
        has_jump=jump_num>0
        
        self.DJ_=np.zeros(shape=(n_paths,n_steps))
        if np.any(has_jump): 
            jump_means=jump_num[has_jump]*mu_J 
            jump_stds=np.sqrt(jump_num[has_jump])*sig_J 
            self.DJ_[has_jump]=rng.normal(loc=jump_means, scale=jump_stds)
        
        if verbose: 
            print("Jump data created. ")
        return self
    
    def draw_paths(self, figsize: tuple=(12,6), time_len: float=1, n_steps: int=252): 
        plt.figure(figsize=figsize) 
        timeline=np.linspace(start=0, stop=time_len, num=n_steps)
        if not hasattr(self, "S_"): 
            ValueError("Stock paths does not exist, please generate before drawing. ") 
        for path in self.S_: 
            plt.plot(timeline,path)
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock value")
        plt.title("Risk free Bates model simulation")
        plt.show()