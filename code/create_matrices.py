create_matrices.py

import os
import glob
import ast
import math
import argparse
import pandas as pd
import numpy as np
import scipy as sp
import random
from scipy import stats
import scipy.io


import seaborn as sea
import matplotlib.pyplot as plt ; plt.ion()
import matplotlib.patches as mpl_patches
from matplotlib import rcParams
from matplotlib import figure 


home = os.path.expanduser('~')

data_dir = os.path.join(home, 'Desktop', 'rest_shen268')

#change to local directory
subj_df = pd.read_csv(data_dir + '/*.csv')




#determine quality ssubjects
sleep_subs = subj_df['subs']
rl_df = subj_df.loc[subj_df['runs'] == 'rest1_RL']
rl_subs = rl_df.loc[rl_df['FD_mean'] <= .15, 'subs']
rl_subs = rl_subs.to_numpy()


lr_df = subj_df.loc[subj_df['runs'] == 'rest2_LR']
lr_subs = lr_df.loc[lr_df['FD_mean'] <= .15, 'subs']
lr_subs = lr_subs.to_numpy()
lr_subs = lr_subs[lr_subs != 160931]
lr_subs = lr_subs[lr_subs != 351938]
lr_subs = lr_subs[lr_subs != 462139]





#load in timeseries
pt_list = []
for i in lr_subs:
    fn = data_dir + '/ptseries_shen_{}_RL.mat'.format(i, run)
    mat = scipy.io.loadmat(fn, struct_as_record=False)
    shen = mat['hcp_shen']
    pt_list.append(shen)




pt_list2 = []
for i in lr_subs:
    fn = data_dir + '/ptseries_shen_{}_LR.mat'.format(i, run)
    mat = scipy.io.loadmat(fn, struct_as_record=False)
    shen = mat['hcp_shen']
    pt_list2.append(shen)



matrix_dir = os.path.join(home, 'Desktop', 'rest matrices', 'LR')


'''
lr_subs2 = []
for i in lr_subs:
    a = str(i)
    b = a.split('.')[0]
    lr_subs2.append(b)
    
lr_subs = []
for i in lr_subs2:
    a = int(i)
    lr_subs.append(a)

'''


#fc matrix functions 
fishies = []

def apply_fisher_z_to_fc(fc):
    fz = np.arctanh(fc)
    np.fill_diagonal(fz, 4.0000)
    fz  = np.round(fz, 5)
    return   fz


def calc_fc(ts):
    num_nodes = ts.shape[0]
    fc = np.full([num_nodes, num_nodes], np.nan)
    for i in range(num_nodes):
        i_ts  = ts[i, :]
        for j in range(num_nodes):
            j_ts  = ts[j, :]
            if np.isnan(j_ts).any() or  np.isnan(i_ts).any():
                fc[i, j] = np.nan
            else:
                fc[i, j] = stats.pearsonr(i_ts, j_ts)[0]
    return  apply_fisher_z_to_fc(fc)




#make fc matrices and send them to directory
for i in range(0, len(pt_list)):
	calc_fc(pt_list[i])
	sub = rl_subs[i]
	np.savetxt(fname=matrix_dir + '/RL/{}.csv'.format(sub), X=calc_fc(pt_list[i]))




for i in range(0, len(pt_list2)):
    calc_fc(pt_list2[i])
    sub = lr_subs[i]
    np.savetxt(fname= matrix_dir + '/LR/{}.csv'.format(sub), X=calc_fc(pt_list2[i]))



