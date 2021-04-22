from dataclasses import dataclass 
import numpy as np
import math
from matplotlib import pyplot as plt
import period_approximant

'''
Code to compute the trajectory of a simple pendulum (starting from rest, if
the approximant is used) using the method of asymptotic approximants. 
'''

#This, as a global variable, gets the method with which the period is computed
PERIOD_FUNCTION = period_approximant.get_af(20)
CEI_FUNCTION = period_approximant.get_complete_ellitpic_integral_appoximant_funct(20)

@dataclass
class Conditions:
    '''
    A dataclass to house the conditions of a pendulum problem
    '''
    theta0: float = np.deg2rad(170) #intial angle. 
    omega0: float = 0 #intial angular velocity. Must be 0 for some functions. 
    l: float = 1 #length of the pendulum string
    g: float = 9.8 #gravitational field
    TT: float = 1.3 #how far out in time to plot the results to
    dt: float = .0001 #timescale used for Euler's method numerical solution
    plotresrat: int = 2000 #how often numerical points are included in plot
    N: int = 5 #what order to compute the series and approximant to. 
    
def numerical_solve(conditions):
    '''
    Uses Euler's method (the idea being to use it with very small timestep)
    to get the exact numerical solution to the problem

    Parameters
    ----------
    conditions : Conditions
        The conditions of the problem to solve

    Returns
    -------
    times : Numpy Array
        the progression of times values in the result
    thetas : Numpy Array
        the progression on angles in the result
        
    '''
    #since we're dealing with a 1-dimensional ODE on a compact domain, Euler's
    #method will suffice with overly small timestep to obtain effective exact 
    #solutions
    dt = conditions.dt
    times = np.linspace(0, conditions.TT, math.ceil(conditions.TT/dt))
    thetas = np.copy(times)
    omegas = np.copy(times)
    kappa = conditions.g / conditions.l
    #add in intiials conditions
    thetas[0] = conditions.theta0
    omegas[0] = conditions.omega0
    #actually do the euler's method. 
    for n in range(len(times)-1):
        thetas[n+1] = thetas[n] + dt * omegas[n]
        omegas[n+1] = omegas[n] - dt * kappa * np.sin(thetas[n])
    return times, thetas

def numerical_plot(conditions):
    '''
    Plots the numerical solution on the plot

    Parameters
    ----------
    conditions : Conditions
        Conditions of the problem to plot.

    '''
    times, angles = numerical_solve(conditions)
    plot_times = times[::conditions.plotresrat]
    plot_angles = angles[::conditions.plotresrat]
    plt.plot(plot_times, plot_angles, 'k.')
    
def get_series(conditions):
    '''
    Returns the coefficients of the Taylors series solution to the pendulum 
    problem

    Parameters
    ----------
    conditions : Conditions
        The conditions with which to compute the series solution.

    Returns
    -------
    a : Numpy Array
        Array where a[k] gives the coefficient of t^k in the solution.

    '''
    #these are in equations (#). Initialize the solution, and its sine and 
    #cosine to be filled with zeros. 
    a = np.zeros(conditions.N + 1)
    C = np.zeros(conditions.N + 1) 
    S = np.zeros(conditions.N + 1)
    #set the initial conditions 
    a[0] = conditions.theta0
    a[1] = conditions.omega0;
    kappa=conditions.g/conditions.l;
    C[0] = np.cos(a[0]) 
    S[0] = np.sin(a[0]);
    #now apply the key equation (#)
    for n in range(conditions.N - 1):
        C[n+1] = -sum((k+1)*a[k+1]*S[n-k] for k in range(n+1))/(n+1)
        S[n+1] = sum((k+1)*a[k+1]*C[n-k] for k in range(n+1))/(n+1)
        a[n+2] = -kappa*S[n]/((n+1)*(n+2)) 
    #and the a is our result
    return a    

def get_series_function(conditions):
    '''
    Returns a function that will compute the series solution value of theta
    for a given time t

    Parameters
    ----------
    conditions : Conditions
        Conditions with which to compute the series solution.

    Returns
    -------
    Python Function
        Function giving the series value for a given time.

    '''
    coeffs = get_series(conditions)
    def theta(t):
        if t == 0:
            return coeffs[0]
        return sum(a*(t**i) for (i, a) in enumerate(coeffs))
    return theta

def get_roc(conditions):
    assert(conditions.omega0 == 0)
    m = math.fabs(np.sin(conditions.theta0/2))
    kappa = conditions.g/conditions.l;
    R = np.sqrt(((CEI_FUNCTION(m**2))**2+(CEI_FUNCTION((1-m)**2))**2)/kappa)
    return R

def series_plot(conditions, wroc = True):
    '''
    Plots the power series solution on the plot

    Parameters
    ----------
    conditions : Conditions
        The conditions of the problem to plot
        
    '''
    #get the raw points. Note that this functional approach can be spread up by
    #broadcasting, but we retain it for aid of use at individual points, which
    #is an advantage of the method of asymptotic approximants
    wroc = (conditions.omega0 == 0)
    series_sol = get_series_function(conditions)
    times = np.linspace(0, conditions.TT, math.ceil(conditions.TT/conditions.dt))
    thetas = np.array([series_sol(t) for t in times])
    #if it's part of the from rest case, cut it off at the obvious point
    if wroc:
        good_indices = list(filter(lambda x: math.fabs(thetas[x]) < 1.1 * conditions.theta0, 
                              range(len(times))))
        times = [times[gi] for gi in good_indices]
        thetas = [thetas[gi] for gi in good_indices]
    #add the radius of convergence line
    if wroc:
        y_vals = np.linspace(0, conditions.theta0)
        x_vals = [get_roc(conditions)] * len(y_vals)
    plt.plot(times, thetas, 'g')
    plt.plot(x_vals, y_vals, 'k')   

def get_quarter_period(conditions):
    '''
    Returns the period of the associated problem. Just a wrapper to scale
    appropriately

    Parameters
    ----------
    conditions : Conditions
        Conditions of the pendulum to find the period of.

    Returns
    -------
    T : float
        The period.

    '''
    assert(conditions.omega0 == 0)
    kappa = conditions.g/conditions.l;
    T = .5 * np.pi * PERIOD_FUNCTION(conditions.theta0) / np.sqrt(kappa)
    return T
    
def get_approx_coeffs(conditions):
    '''
    Returns the coefficients of the approximant. 

    Parameters
    ----------
    conditions : Condtions
        The conditions of the problem for which to compute

    Returns
    -------
    a_hat : Numpy Array
        A numpy array with the coefficients

    '''
    assert(conditions.omega0==0)
    T_4 = get_quarter_period(conditions)
    #get the series solution
    a = get_series(conditions)
    b = np.copy(a)
    #modify the early terms to subtract jet at T/4
    theta_dot_b = np.sqrt(2*conditions.g*(1-np.cos(a[0]))/conditions.l)
    b[0] = a[0] - T_4 * theta_dot_b
    b[1] = a[1] + theta_dot_b
    #next, we actually compute the caychy's product rule
    a_hat = np.zeros(conditions.N+1)
    for n in range(len(a_hat)):
        a_hat[n] = sum(b[n-k] * (k+1) * (T_4 ** (-k-2)) for k in range(n+1))
    return a_hat
    
def get_approximant(conditions):
    '''
    Returns a function that computes the approximant for a given time

    Parameters
    ----------
    conditions : Conditions
        The conditions of the problem for which to compute

    Returns
    -------
    Python Function
        Gives the resulting approximant value of the angle for any time

    '''
    assert(conditions.omega0==0)
    T_4 = get_quarter_period(conditions)
    coeffs = get_approx_coeffs(conditions)
    theta_dot_b = np.sqrt(2*conditions.g*(1-np.cos(conditions.theta0))/conditions.l)
    def theta(t):
        if t == 0:
            return conditions.theta0
        return (T_4 - t) * theta_dot_b + (T_4 - t)**2 * sum(a * (t ** i) for (i, a) in enumerate(coeffs))
    return theta

def approximant_plot(conditions):
    '''
    Plots the approximant olution on the plot

    Parameters
    ----------
    conditions : Conditions
        The conditions of the problem to plot
        
    '''
    #Same spiel about how efficiency could be increased with broadcasting if 
    #you know a priori, like you do in a plot, that you're going to want many 
    #specific values
    assert(conditions.omega0 == 0)
    approximant = get_approximant(conditions)
    times = np.linspace(0, conditions.TT, math.ceil(conditions.TT/conditions.dt))
    thetas = np.array([approximant(t) for t in times])
    plt.plot(times, thetas, 'r')
    
def plot_all(conditions):
    '''
    Plots all of the numerical solution, series solution, and apporximant 
    solution for the given conditions

    Parameters
    ----------
    conditions : Conditions
        The conditions to plot.

    '''
    numerical_plot(conditions)
    series_plot(conditions)
    approximant_plot(conditions)
    
if __name__ == "__main__":
    m_conditions = Conditions(N = 10)
    plot_all(m_conditions)
        
    
    