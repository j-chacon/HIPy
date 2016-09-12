# -*- coding: utf-8 -*-
"""
Performance Functions

Reference to the methods:

P Krause, DP Boyle, F Bäse. Comparison of different efficiency criteria for 
hydrological model assessment. Advances in Geosciences.5, 89-97, EGU 2005.

x - calculated value
y - recorded value
q - Quality tag (0-1)
"""
import numpy as np
import scipy.stats as stt
# constants import 

# Or something similar,.... like importing a module of contants 
#or something
ERROR_CODE = -9999

def DataValid(x,y,q):
    """
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    
    if len(x) != len(y):
        raise NameError('Calculated and recorded series length do not match len(x) = {0}, len(y) = {1}'.format(len(x), len(y)))
        return ERROR_CODE
        
    if len(q) != len(y):
        raise NameError('Quality tags vector length do not match with measurements')
        return ERROR_CODE
    
    if max(q) == 0:
        raise NameError('Totally unreliable data, check quality tags')
        return ERROR_CODE
    
    if np.amin(q) < 0:
        raise NameError('Quality tags cannot be negative')
        return ERROR_CODE
    
    if np.amax(q) > 1:
        raise NameError('Quality tags cannot be greater than 1')
        return ERROR_CODE
        
    try:
        np.sum(x)
    except ValueError:
        raise NameError('Calculated data might contain non-numerical values')
        return ERROR_CODE
        
    try:
        np.sum(y)
    except ValueError:
        raise NameError('Recorded data might contain non-numerical values')
        return ERROR_CODE
    
    try:   
        np.sum(q)
    except ValueError:
        raise NameError('Quality tags might contain non-numerical values')
        return ERROR_CODE
        
    x = np.array(x)
    y = np.array(y)    
    q = np.array(q)
    return x,y,q

def rmse(x,y,q='def'):
    """
    ========================
    Root mean squarred error
    ========================
    
    Parameters
    ----------    
    x : calculated value
    y : recorded value
    q : Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE    
        
    Erro = np.square(x-y)*q    
    cdef float F = np.sqrt(1.*np.sum(Erro)/(np.sum(q)))
    return F
    
def nrmse(x,y,q='def'):
    """
    ===================================
    Normalised root mean squarred error
    ===================================
    
    Parameters
    ----------
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE    
        
    Erro = np.square(x-y)*q    
    cdef float F = (np.sqrt(1.*np.sum(Erro)/(np.sum(q))))/(np.amax(y)-np.amin(y))
    return F

def rsr(x,y,q='def'):
    """
    ===================================
    Ratio of RMSE to standard deviation of observations
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE    
        
    Erro = np.square(x-y)*q    
    Erro2 = np.square(y-np.average(y))*q
    cdef float F = np.sqrt(1.*np.sum(Erro)/(np.sum(q)))/np.sqrt(1.*np.sum(Erro2))
    return F
    
def rsd(x,y,q='def'):
    """
    ===================================
    Ratio od Standard Deviations
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        

    cdef float F = np.std(x)/np.std(y)
    return F

def bias(x,y,q='def'):
    """
    ====
    Bias
    ====
    
    Parameters
    ----------
    Performance Functions
    x : calculated value
    y : recorded value
    q : Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    
    cdef float F = np.sum(np.subtract(x,y)*q)/np.sum(q)
    return F

def pbias(x,y,q='def'):
    """
    ===================================
    Percentage Bias
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    cdef float F = np.sum(np.subtract(x,y)*q)/np.sum(y*q)
    return F

def mae(x,y,q='def'):
    """
    ===================================
    Mean Averaged Error
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))    
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    cdef float F = (1./sum(q))*np.abs(np.sum(x-y))
    return F
    
def mse(x,y,q='def'):
    """
    ===================================
    Mean Standard Error
    ===================================
    
    Parameters
    ----------
    ===================
    Mean Squarred Error
    ===================
    
    Parameter
    ---------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))    
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE    
    Erro = np.square(x-y)*q    
    cdef float F = np.sum(Erro)/np.sum(q)
    return F   
    
def perc_vol(x,y,q='def'):
    """
    =================
    Percentage volume
    =================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   
    
    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    cdef float sy = np.sum(y)
    cdef float F = (np.sum(x)-sy)/sy
    return F

def nse(x,y,q='def',j=2.0):
    """
    =========================
    Nash-Sutcliffe Efficiency
    =========================

    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    j - exponent to modify the inflation of the variance (standard NSE j=2)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    a = np.sum(np.power(x-y,j)*q)
    b = np.sum(np.power(y-np.average(y),j)*q)
    cdef float F = 1.0 - a/b
    return F

def lnse(x,y,q='def',j=2.0):
    """
    ===================================
    Log-Nash-Sutcliffe Efficiency
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    j - exponent to modify the inflation of the variance (standard NSE j=2)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    
    x = np.log(x)
    y = np.log(y)
    
    a = np.sum(np.power(x-y,j)*q)
    b = np.sum(np.power(y-np.average(y),j)*q)
    cdef float F = 1.0 - a/b
    return F

def rnse(x,y,q='def',j=2.0):
    """
    ===================================
    Ranked Nash-Sutcliffe Efficiency
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    j - exponent to modify the inflation of the variance (standard NSE j=2)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    a = np.sum(np.power(np.subtract(x,y)/np.average(y),j)*q)
    b = np.sum(np.power(np.subtract(y,np.average(y))/np.average(y),j)*q)
    cdef float F = 1.0 - a/b
    return F

def ioa(x,y,q='def',j=2.0):
    """
    ===================================
    Index of Agreement
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    a = np.sum(np.power(np.subtract(x,y)*q,j))
    b = np.sum(np.power(np.abs(np.subtract(x,y)*q)+np.abs(np.subtract(y,np.average(y))*q),j))
    cdef float F = 1.0 - a/b
    return F

def rioa(x,y,q='def',j=2.0):
    """
    ===================================
    Ratio Index of Agreement
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    a = np.sum(np.power(np.subtract(x,y)*q/y,j))
    b = np.sum(np.power(np.abs(np.subtract(x,y)*q/y)+np.abs(np.subtract(y,np.average(y))*q/y),j))
    cdef float F = 1.0 - a/b
    return F
    
def pearsonr(x,y,q='def'):
    """
    ===================================
    Pearson's R
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    return stt.pearsonr(x,y)[0]

def spearmanr(x,y,q='def'):
    """
    ===================================
    Spearman's R
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    return stt.spearmanr(x,y)[0]

def detcoeff(x,y,q='def'):
    """
    ===================================
    Determination coefficient
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
        
    return np.power(stt.pearsonr(x,y)[0],2)

def cop(x,y,q='def', lag=1):
    """
    ===================================
    Nash-Sutcliffe of lagged series
    ===================================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    
    a = np.sum(np.power(np.subtract(y[lag:]-x[lag:])*q,2))
    b = np.sum(np.power(np.subtract(y[lag:]-y[:-lag])*q,2))
    cdef float F = 1.0 - a/b
    return F

def kge(x, y, q='def', s=[1,1,1]):
    """
    ======================
    Klimt-Gupta Efficiency
    ======================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    s - inflation of different parameters (see:Gupta, 2009)
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    
    if min(q) == 0:
        for i in xrange(0,len(x)):
            CleanDum = []
            if q[i] == 0:
                CleanDum.append(i)
        
        x = np.delete(x,CleanDum)
        y = np.delete(y,CleanDum)
            
    r = stt.pearsonr(x,y)[0]
    alp = np.std(x)/np.std(y)
    bet = (np.mean(x)-np.mean(y))/np.std(y)
    ed = np.sqrt((s[0]*(r-1))**2+(s[1]*(alp-1))**2+(s[2]*(bet-1))**2)
    kge_out = 1-ed
    return kge_out
    

def willmottd(x, y, q='def', c=2):
    """
    ======================
    Willmott's D
    ======================
    
    Parameters
    ----------
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    
    ref: http://onlinelibrary.wiley.com/doi/10.1002/joc.2419/full
    """
    if q is 'def':
        q = np.ones(len(y))   

    x,y,q = DataValid(x,y,q)
    if min(x) == ERROR_CODE:
        return ERROR_CODE
    
    if min(q) == 0:
        for i in xrange(0,len(x)):
            CleanDum = []
            if q[i] == 0:
                CleanDum.append(i)
        
        x = np.delete(x,CleanDum)
        y = np.delete(y,CleanDum)
        
    y_mean = np.average(y)
    tot_error = np.sum(np.abs(x-y))
    tot_dev = np.sum(np.abs(y-y_mean))
    
    if tot_error <= c*tot_dev:
        w_d = 1 - (tot_error/(c*tot_dev))
    else:
        w_d = ((c*tot_dev)/tot_error) - 1

    return w_d