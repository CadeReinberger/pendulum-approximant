import approximant 
import period_approximant
import pandas as pd
import os
import math
import numpy as np
import sys

def clear_data_folder():
    dfpath = "data/" #Enter your path here
    for root, dirs, files in os.walk(dfpath):
        for file in files:
            os.remove(os.path.join(root, file))

def get_all_trajectory_data(conditions):
    num_times, num_thetas = approximant.numerical_solve(conditions)
    series_funct = approximant.get_series_function(conditions)
    approximant_funct = approximant.get_approximant(conditions)
    series_thetas = [series_funct(t) for t in num_times]
    approximant_thetas = [approximant_funct(t) for t in num_times]
    return num_times, num_thetas, series_thetas, approximant_thetas

def get_trajectory_dataframe(conditions):
    nts, nths, sths, aths = get_all_trajectory_data(conditions)
    df_dict = {'times' : nts, 'num_thetas' : nths,
               'series_thetas' : sths, 'approximant_thetas' : aths}
    m_df = pd.DataFrame.from_dict(df_dict)
    return m_df

def save_trajectory_data(N):
    conditions = approximant.Conditions(N=N)
    get_trajectory_dataframe(conditions).to_csv('data/data_' + str(conditions.N) + '.csv')
    
def get_abs_traj_error_data(conditions):
    nts, nths, sths, aths = get_all_trajectory_data(conditions)
    series_errors = [math.fabs(sths[ind] - nths[ind]) for ind in range(len(nts))]
    approximant_errors = [math.fabs(aths[ind] - nths[ind]) for ind in range(len(nts))]
    return nts, series_errors, approximant_errors

def get_abs_traj_error_dataframe(conditions):
    nts, series_errors, approximant_errors = get_abs_traj_error_data(conditions)
    df_dict = {'ts' : nts, 'series_errors' : series_errors,
               'approximant_errors' : approximant_errors}
    m_df = pd.DataFrame.from_dict(df_dict)
    return m_df

def save_abs_traj_error_data(N):
    conditions = approximant.Conditions(N=N)
    get_abs_traj_error_dataframe(conditions).to_csv('data/data_' + str(conditions.N) + '.csv')
    
def get_rel_traj_error_data(conditions):
    nts, nths, sths, aths = get_all_trajectory_data(conditions)
    series_errors = [math.fabs((sths[ind] - nths[ind]) / nths[ind]) for ind in range(len(nts))]
    approximant_errors = [math.fabs((aths[ind] - nths[ind]) / nths[ind]) for ind in range(len(nts))]
    return nts, series_errors, approximant_errors

def get_rel_traj_error_dataframe(conditions):
    nts, series_errors, approximant_errors = get_rel_traj_error_data(conditions)
    df_dict = {'ts' : nts, 'series_errors' : series_errors,
               'approximant_errors' : approximant_errors}
    m_df = pd.DataFrame.from_dict(df_dict)
    return m_df

def save_rel_traj_error_data(N):
    conditions = approximant.Conditions(N=N)
    get_rel_traj_error_dataframe(conditions).to_csv('data/data_' + str(conditions.N) + '.csv')

def get_all_period_data(N, theta_max = np.deg2rad(179.9), num = 1000):
    thetas = np.linspace(0, theta_max, num=num)
    ef = period_approximant.get_exact_funct()
    exact_ts = [ef(t) for t in thetas]
    tf = period_approximant.get_taylor_funct(N)
    t_ts = [tf(t) for t in thetas]
    aaf = period_approximant.get_af(N)
    aa_ts = [aaf(t) for t in thetas]
    return thetas, exact_ts, t_ts, aa_ts

def get_period_dataframe(N):
    thetas, ets, sts, ats = get_all_period_data(N)
    df_dict = {'thetas' : thetas, 'exact_periods' : ets,
               'series_periods' : sts, 'approximant_periods' : ats}
    m_df = pd.DataFrame.from_dict(df_dict)
    return m_df

def save_period_data(N):
    get_period_dataframe(N).to_csv('data/data_' + str(N) + '.csv')
    
def get_abs_period_error_data(N):
    thetas, ets, sts, ats = get_all_period_data(N)
    series_errors = [math.fabs((sts[ind] - ets[ind]) / ets[ind]) for ind in range(len(thetas))]
    approximant_errors = [math.fabs((ats[ind] - ets[ind]) / ets[ind]) for ind in range(len(thetas))]
    return thetas, series_errors, approximant_errors

def get_abs_period_error_dataframe(N):
    thetas, series_errors, approximant_errors = get_abs_period_error_data(N)
    df_dict = {'ts' : thetas, 'series_errors' : series_errors,
               'approximant_errors' : approximant_errors}
    m_df = pd.DataFrame.from_dict(df_dict)
    return m_df

def save_abs_period_error_data(N):
    get_abs_period_error_dataframe(N).to_csv('data/data_' + str(N) + '.csv')
    
    
def get_rel_period_error_data(N):
    thetas, ets, sts, ats = get_all_period_data(N)
    series_errors = [math.fabs((sts[ind] - ets[ind]) / ets[ind]) for ind in range(len(thetas))]
    approximant_errors = [math.fabs((ats[ind] - ets[ind]) / ets[ind]) for ind in range(len(thetas))]
    return thetas, series_errors, approximant_errors

def get_rel_period_error_dataframe(N):
    thetas, series_errors, approximant_errors = get_rel_period_error_data(N)
    df_dict = {'ts' : thetas, 'series_errors' : series_errors,
               'approximant_errors' : approximant_errors}
    m_df = pd.DataFrame.from_dict(df_dict)
    return m_df

def save_rel_period_error_data(N):
    get_rel_period_error_dataframe(N).to_csv('data/data_' + str(N) + '.csv')
    
operator_map = {('traj', 'norm') : save_trajectory_data,
                ('traj', 'abs') : save_abs_traj_error_data,
                ('traj', 'rel') : save_rel_traj_error_data,
                ('per', 'norm') : save_period_data,
                ('per', 'abs') : save_abs_period_error_data,
                ('per', 'rel') : save_rel_period_error_data}
    
if __name__ == '__main__':
    assert(sys.argv[1].strip().lower() in ('traj', 'per'))
    assert(sys.argv[2].strip().lower() in ('norm', 'abs', 'rel'))
    clear_data_folder()
    params = (sys.argv[1].strip().lower(), sys.argv[2].strip().lower())
    save_funct = operator_map[params]
    ns = [int(i) for i in sys.argv[3:]]
    for n in ns:
        save_funct(n)
