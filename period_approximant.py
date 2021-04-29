import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom

'''
Code to compute the period of a simple pendulum (starting from rest) using the
method of asymptotic approximants. 

Throughout, we make the simplifying assumption that l = g. If this assumption
does not hold, the result need only be multiplied by sqrt(l/g). 

We also make the assumption that the initial angular velocity is 0. This too 
can be easily adjusted b simply first computing the angle at which the angular
velocity is 0 with the conservation of energy. 
'''

def get_taylor_series(N):
    '''
    Returns the Taylor series to n-th order of the complete ellitpic intrgral
    of the first kind expanded about 0. 

    Parameters
    ----------
    N : INT
        Order to take the series expansion to. 

    Returns
    -------
    Numpy array
        Taylor series expansion to order N.

    '''
    res = np.zeros(N+1)
    res[0] = 1
    for m in range(1,N//2+1):
        #more efficient recursive computation of out central binomial
        #coefficient expression with double factorials
        res[2*m-1] = 0
        res[2*m] = res[2*m-2] * ((2*m-1)/(2*m))**2
    return np.array(res)
    
def get_taylor_funct(N):
    '''
    Returns a function that computes the N-th order taylor expansion for the

    Parameters
    ----------
    N : INT
        Order to take the series expansion to. 

    Returns
    -------
    Python Function 
        The function will take in a float theta0 and output a float period

    '''
    series = get_taylor_series(N)
    def tf(t0):
        s = np.sin(t0/2)
        return sum(a*(s**i) for (i, a) in enumerate(series)) if s != 0 else 1
    return tf
    

def get_amgm_funct(N):
    '''
    State of the art non-analytic numerical method to n-th order. Uses the 
    quadratically-convergent method Carvahalles and Suppes of computing the
    complete elliptic integral with fixed point iteration on the arithmetic-
    geometric mean. 

    Parameters
    ----------
    N : INT
        Number of times to apply the fixed point procedure.

    Returns
    -------
    approx_period : Python function 
        Function to compute the period with this method on N iterations.

    '''
    amgm_iter = lambda t : (.5*(t[0] + t[1]), np.sqrt(t[0]*t[1]))
    def approx_period(t0):
        cur = (1, np.cos(t0/2))
        for _ in range(N+1):
            cur = amgm_iter(cur) 
        return cur[0] ** -1
    return approx_period

def get_exact_funct():
    '''
    Used as an exact solution. Uses the state of the art purely numerical 
    solution to sufficiently high order. 

    Returns
    -------
    Function giving the exact solution for the period to within roughly 
    neglible error. 

    '''
    return get_amgm_funct(100) #don't need this man, but alas

def get_coeff(k):
    '''
    Returns the n-th coefficient of the approximant. This could be sped up by 
    using a recursive implementation such as the one given in the paper, but 
    this equation directly mirror equaiton (#) in the paper. 

    Parameters
    ----------
    k : INT
        The power of m the coefficient multiplies in the expansion

    Returns
    -------
    FLOAT
        The coefficient on the m^k term. 

    '''
    #this is directly equaiton (#) in the paper
    return (((np.pi/2)*(binom(2*k, k)**2)*(16**(-k)))- (1/(2*k+1)))

def get_complete_ellitpic_integral_appoximant_funct(N):
    '''
    Returns a function that gives the approximant for the complete elliptic 
    integral to order N. 

    Parameters
    ----------
    N : INT
        The order at which to truncate the sereies term in the approximant

    Returns
    -------
    Python Function
        Returns a function that gives the approximant

    '''
    def get(x):
        if x == 0:
            return np.pi/2
        tot = sum(get_coeff(k) * (x**k) for k in range(0, N+1))
        return np.arctanh(np.sqrt(x))/np.sqrt(x) + tot
    return get

def get_af(N):
    '''
    A wrapper on get_complete_ellitpic_integral_appoximant_funct(N) that gives
    a function that computes the same approximant to order N but uses it
    directly to compute the pendulum period. 

    Parameters
    ----------
    N : INT
        The order at which we truncate the approximant. 

    Returns
    -------
    Python Function
        The approximation for the pendulum period to order N. 
    '''
    ceiaf = get_complete_ellitpic_integral_appoximant_funct(N)
    return lambda theta: (2/np.pi)*ceiaf(np.sin(theta/2)**2)

def plot_all(N, theta_max = np.deg2rad(179.9), num = 1000):
    '''
    Makes and displays a matplotlib plot of the exact answer, the series 
    solution, and the approximant to order N. 

    Parameters
    ----------
    N : INT
        The order to use for the series expansion and approximant. 
    theta_max : FLOAT, optional
        the maximum angle to plot up to. The value is infinite at or above
        180 degrees. The default is np.deg2rad(179.9).
    num : INT, optional
        The number of point to use in the plots. The default is 1000.

    Returns
    -------
    None.

    '''
    thetas = np.linspace(0, theta_max, num=num)
    #first, the exact solution
    ef = get_exact_funct()
    exact_Ts = [ef(t) for t in thetas]
    #next, the state of the art solution 
    # sotaf = get_amgm_funct(N)
    # sota_Ts = [sotaf(t) for t in thetas]
    #next, the taylor solution
    tf = get_taylor_funct(N)
    t_Ts = [tf(t) for t in thetas]
    #now, the approximant
    aaf = get_af(N)
    aa_Ts = [aaf(t) for t in thetas]
    #finally, plot them all
    plt.plot(thetas, t_Ts, color = 'g')
    plt.plot(thetas, aa_Ts, color = 'r')
    plt.plot(thetas, exact_Ts, color = 'k', alpha = .7)

#eighth order apporximaitons are a default pretty plot. 
if __name__ == "__main__":
    plot_all(5)