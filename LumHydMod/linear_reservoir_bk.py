# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:31:08 2015

@author: chaco3

Tank Model

Implemented By Juan Chacon
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


INITIAL_Q = 1.0
INITIAL_PARAM = [0.01, 1.0]
PARAM_BND = ((0.0001, 1.1),
             (0.7, 1.7))

#%%
def _step(prec_step, evap_step, q_old, param, extra_param):
    '''
    prec mm/hr
    evap = mm/hr    
    '''
    # Transformation of precipitation into inflow (m³/hr)
    inp = np.max([(prec_step*param[1] - evap_step)*extra_param[1]*1000.0, 0])
    
    # Get discharge in m³/hr
    q_sim = (q_old*3600.0)*np.exp(-param[0]*extra_param[0]) + inp*(1.0 - np.exp(-param[0]*extra_param[0]))
    
    # Change to m³/s    
    q_sim = q_sim/3600.0
    return q_sim

def simulate(prec, evap, param, extra_param):
    '''

    '''
    q = [INITIAL_Q, ]

    for i in xrange(len(prec)):
        step_res = _step(prec[i], evap[i], q[i], param, extra_param)
        q.append(step_res)

    return q

def calibrate(prec, evap, extra_param, q_rec, verbose=False):

    def mod_wrap(param_cal):
        q_sim = simulate(prec[:-1], evap[:-1], param_cal, extra_param)
        try:
            perf_fun = -NSE(q_sim, q_rec)
        except:
            perf_fun = 9999

        if verbose: print -perf_fun
        return perf_fun

    cal_res = opt.minimize(mod_wrap, INITIAL_PARAM, bounds=PARAM_BND,
                           method='L-BFGS-B')

    return cal_res.x, cal_res.fun

def NSE(x,y,q='def',j=2.0):
    """
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    j - exponent to modify the inflation of the variance (standard NSE j=2)
    """
    x = np.array(x)
    y = np.array(y)
    a = np.sum(np.power(x-y,j))
    b = np.sum(np.power(y-np.average(y),j))
    F = 1.0 - a/b
    return F

