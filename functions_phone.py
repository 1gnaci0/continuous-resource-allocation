# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:44:03 2018

@author: ignacio
"""

import numpy as np
from scipy.optimize import fsolve
from scipy import integrate

#%%
def g(eta): 
    if abs(eta) < 1e-4:
        # Defined to be continuous in eta = 0 
        # Fith order Taylor series at eta ~ 0 
        return 0.5 + (1./12.)*eta - (1./720)*eta**3 + (1./30240)*eta**5
    return (np.exp(eta)/(np.exp(eta)-1.)) - (1./eta)

#%%
def function_to_fit(x,eta,smax,smin):
    '''
    x is the intensity measured (scalar)
    eta, smax, and smin are to be determined extrinsically   
    '''    
    if abs(eta) < 1e-4:
        # Defined to be continuous in eta = 0 
        # Second order Taylor series at eta ~ 0 
        t = (smax - x) / (smax - smin)
        return t + (1./2.)*(t -1.)*t*eta + (1./12)*(2*t**2 -3*t +1)*eta**2
    a = (eta*smax)/(smax-smin)
    b = eta/(smax - smin)    
    return (np.exp(a - b*x) - 1.) / (np.exp(eta) - 1.)
#%%
def y(t,as_discrete=False):
    unique = np.unique(t)
    N = len(unique) #+1#+ 1.
    y = [] 
    i = 1.
    for z in sorted(unique,reverse= not as_discrete):
       ocurrences = range(t.count(z))
       for ocurrence in ocurrences:
           y.append(i / N)
       i += 1.      
    return y
#%%
    ### INTEGRATION METHODS ### 
def integrand_leg(x,R,L):
    # Integrand for Gauss_Legendre quadrature.  See SI Eq (30) and Methods
    if abs(x) < 1e-6:
	# Taylor expansion of quotient up to fourth order at 0
	# It prevents bad behaviours near zero 
        ev_log = L*np.log(1. + (x/2.) + (x**2 / 12.) - (x**4 / 720.)) -x*R
        return np.exp(ev_log)
    else:	
	# First evaluate log, then exponentiate
        ev_log1 = -x*R + L*np.log((x)/(1. - np.exp(-x)))
        return np.exp(ev_log1)
   
          
def integrand_lag(x,R,L):
    # Integrand for Gauss_Laguerre quadrature.  See SI Eq (30) and Methods
    if abs(x) < 1e-6:    
        # Taylor expansion of quotient up to third order at x=0
        # It prevents bad behaviours near zero (not needed for the cases analysed in the paper)
        ev_log2 = -(L+1.)*np.log(R) +L*np.log(R + (x/2.) + (x**2 / (12.*R)) - (x**4 / (720.*(R**3))))
        return np.exp(ev_log2)     
    else:
        # First evaluate log, then exponentiate
        ev_log3 = -(L+1.)*np.log(R) +L*np.log((x)/(1. - np.exp(-x/R)))
        return np.exp(ev_log3)       
    
def F_t(t,R,L):
    # R is either L1 or (L - L1) See SI Eq(54)

    if t != np.inf:
    #For finite limits of integration use Gauss-Legendre
        return integrate.quad(integrand_leg,args = (R,L), a=0., b=t)[0]
         
    else:
        #For infinite limits of integration use Gauss-Laguerre
        ti, wi, = np.polynomial.laguerre.laggauss(150) 
        f_xi = [integrand_lag(t,R,L) for t in ti]      
        return np.dot(f_xi, wi)
        
def G(t,L1,L):
	# Cumulative distribution of the parameter. See SI Eq (37)
    L2 = L - L1
    F_inf_L1 = F_t(np.inf, L1,L)
    F_inf_L2 = F_t(np.inf, L2,L)
    norm = F_inf_L1 + F_inf_L2
            
    if t < 0.:
        numerator = F_inf_L1 - F_t(-t,L1,L) 
    else:
        numerator = F_inf_L1 + F_t(t,L2,L)
    
    result = numerator / norm  
    
    return result

def eta_confidence_interval(L1, L, d=0.05, initial=1.,tol=1e-6):
    t1 = fsolve(lambda x: G(x,L1,L) - d/2, x0=initial - 1e-1,
                xtol=tol)[0]     # Looks for t1 starting eps to the left of the mode if intital = eta
    t2 = fsolve(lambda x: G(x,L1,L) - (1-d/2),x0=initial + 1e-1,
                xtol=tol)[0]     # Looks for t2 starting eps to the right of the mode if initial = eta
    return (t1,t2)

#%%
 
def d_ref(ego, series_of_egos_eta):
    d_ij = 0.
    alters_eta = list(series_of_egos_eta[series_of_egos_eta.index != ego])
    ego_eta = series_of_egos_eta.loc[ego]
    for alter_eta in alters_eta:
        d = np.abs((ego_eta - alter_eta) / ego_eta)
        d_ij += d
    return d_ij /  len(alters_eta)  

#def d_ref(ego, series_of_egos_eta):
#    d_ij = 0.
#    alters_eta = list(series_of_egos_eta[series_of_egos_eta.index != ego])
#    ego_eta = series_of_egos_eta.loc[ego]
#    ref_eta = np.mean(alters_eta)
#    
#    return np.abs((ego_eta - ref_eta) / ego_eta)

def d_ref_v2(ego, series_of_egos_eta):
    out = []
    d_ij = 0.
    alters_eta = list(series_of_egos_eta[series_of_egos_eta.index != ego])
    ego_eta = series_of_egos_eta.loc[ego]
    for ref_eta in alters_eta:
        ref = np.abs((ego_eta - ref_eta) / ego_eta)
        out.append(ref)
    
    return out
#%%
def linear_fit(x,a):
    return a*np.array(x)

#%%
def is_message(a,b,c):
    if (a==b) and c==1:
        return False
    else:
        return True
        
#%%
#########################################
####  FACEBOOK/SOCIOPATTERNS        #####
#########################################

def interaction_is_in_day(seconds, day=1):
    if day == 1:
        interval = [0,86420]
    elif day == 2:
        interval = [86420, 86400*2]
    elif day == 3:
        interval = [86400*2,86420*3]
    else:
        return 'invalid day'
    return (seconds >= interval[0]) and (seconds <= interval[1])


def total_interaction_per_day(data,day=1):
    new = {ego:[] for ego in data}    
    for ego in data:
        temp = []
        for alter in data[ego].keys():
            values = [list_[1] - list_[0] for list_ in data[ego][alter] if interaction_is_in_day(list_[0],day)]
            suma = sum(values)
            if suma != 0:
                temp.append(suma)
        new[ego] += temp
        if len(new[ego]) == 0:
            new[ego] = [0]
    return new

        
def smin_smax(data):
    dic = {ego:() for ego in data}
    for ego in data:
        interactions_day_1 = total_interaction_per_day(data,day=1)[ego]
        interactions_day_2 = total_interaction_per_day(data,day=2)[ego]
        interactions_day_3 = total_interaction_per_day(data,day=3)[ego]
    
        smin = min(interactions_day_1) + min(interactions_day_2) + min(interactions_day_3)
        smax = max(interactions_day_1) + max(interactions_day_2) + max(interactions_day_3)
        dic[ego] = (smin,smax)
    return dic

def total_interaction(data):
    new = {ego:[] for ego in data}
    for ego in data:
        temp = []
        for alter in data[ego].keys():
            values = [list_[1] - list_[0] for list_ in data[ego][alter]]
            suma = sum(values)
            temp.append(suma)
        new[ego] += temp
    return new