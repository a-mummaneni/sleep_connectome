
abcd_rest_cpm.py 

import os
import glob
import ast
import math
import argparse
import pandas as pd
import pingouin as pg
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
from statistics import mean
import warnings
from nilearn.plotting import plot_connectome
import pingouin as pg
import statistics



#load in all files
home = os.path.expanduser('~')

data_dir = os.path.join(home, 'Desktop', 'abcd_rest')

dir1 = os.path.join(home, 'Desktop', 'abcd_rest', 'run01')
dir2 = os.path.join(home, 'Desktop', 'abcd_rest', 'run02')
dir3 = os.path.join(home, 'Desktop', 'abcd_rest', 'run03')
dir4 = os.path.join(home, 'Desktop', 'abcd_rest', 'run04')


dirs1 = sorted(glob.glob(dir1 + '/*.csv'))
dirs2 = sorted(glob.glob(dir2 + '/*.csv'))
dirs3 = sorted(glob.glob(dir3 + '/*.csv'))
dirs4 = sorted(glob.glob(dir4 + '/*.csv'))


sleep_dir = os.path.join(home, 'Desktop', 'fdss')
sleep_df = pd.read_csv(sleep_dir + '/abcd_fbdss01.txt', delimiter = '\t', dtype={'subjectkey' : str})
sleep_df = sleep_df.iloc[1: , :]
sleep_df = sleep_df[sleep_df.eventname == '2_year_follow_up_y_arm_1']


#find id lists
ids1 = [] 

for i in dirs1:
    subject = i.split('/')[6]
    number = subject.split('-')[1]
    sub_id = number.split('_')[0]
    ids1.append(sub_id)


ids2 = [] 

for i in dirs2:
    subject = i.split('/')[6]
    number = subject.split('-')[1]
    sub_id = number.split('_')[0]
    ids2.append(sub_id)



ids3 = [] 

for i in dirs3:
    subject = i.split('/')[6]
    number = subject.split('-')[1]
    sub_id = number.split('_')[0]
    ids3.append(sub_id)


ids4 = [] 

for i in dirs4:
    subject = i.split('/')[6]
    number = subject.split('-')[1]
    sub_id = number.split('_')[0]
    ids4.append(sub_id)

def common_member(a, b):   
    a_set = set(a)
    b_set = set(b)
     
    # check length
    if len(a_set.intersection(b_set)) > 0:
        return(a_set.intersection(b_set)) 
    else:
        return("no common elements")

first = common_member(ids1, ids2)
second = common_member(first, ids3)
subj_ids = common_member(second, ids4)



#find all motion subjects
motion_df = pd.read_csv(sleep_dir + '/release_3_curated_mean_fd_by_run.csv')
motion_df =  motion_df.loc[motion_df['task'] == 'rest']
motion_df = motion_df[motion_df.fd < .15]

motion_df1 = motion_df[motion_df.run == 1]
subs1 = motion_df1['subject']
subs1 = subs1.to_numpy()
subj_ids1 = common_member(subj_ids, subs1)


motion_df2 = motion_df[motion_df.run == 2]
subs2 = motion_df2['subject']
subs2 = subs2.to_numpy()
subj_ids2 = common_member(subj_ids, subs2)


motion_df3 = motion_df[motion_df.run == 3]
subs3 = motion_df3['subject']
subs3 = subs3.to_numpy()
subj_ids3 = common_member(subj_ids, subs3)


motion_df4 = motion_df[motion_df.run == 4]
subs4 = motion_df4['subject']
subs4 = subs4.to_numpy()
subj_ids4 = common_member(subj_ids, subs4)




def read_in_matrices(ids, file_suffix=None, data_dir=data_dir, zscore=False, run=1):
    
    all_fc_data = {}
            
    for subj in ids:
        
        # read it in and make sure it's symmetric and has reasonable dimensions
        tmp = np.loadtxt(data_dir+'/run0{}/sub-{}_ses-baselineYear1Arm1_task-rest_run-0{}_desc-clean_36p_desc-ts_shen_268_desc-atlas_connectivity_despiked.csv'.format(run, subj, run), delimiter=',')
        tmp = tmp[:,1:]
        assert tmp.shape[0]==tmp.shape[1]>1, "Matrix seems to have incorrect dimensions: {}".format(tmp.shape)
        
        # take just the upper triangle and store it in a dictionary
        if ~zscore:
            all_fc_data[subj] = tmp[np.triu_indices_from(tmp, k=1)]
        if zscore:
            all_fc_data[subj] = sp.stats.zscore(tmp[np.triu_indices_from(tmp, k=1)])
        
    # Convert dictionary into dataframe
    all_fc_data = pd.DataFrame.from_dict(all_fc_data, orient='index')
    
    return all_fc_data


fc_data1 = read_in_matrices(subj_ids1, data_dir=data_dir, run=1)
fc_data2 = read_in_matrices(subj_ids2, data_dir=data_dir, run=2)
fc_data3 = read_in_matrices(subj_ids3, data_dir=data_dir, run=3)
fc_data4 = read_in_matrices(subj_ids4, data_dir=data_dir, run=4)
fc_data = pd.concat([fc_data1, fc_data2, fc_data3, fc_data4]).groupby(level=0).mean()
#create fc matrix


#make sub ids just numbers
subs = fc_data.index


sleep_subs = sleep_df['subjectkey']
new_subs = []
for i in sleep_subs:
    new_subs.append(i.replace('_', ''))


sleep_df['subjectkey'] = new_subs


#find minutes slept and fd avs
minutes = []
excluded = []
for i in subs:
    a = sleep_df.loc[sleep_df['subjectkey'] == i, 'fit_ss_sleepperiod_minutes']
    if len(a) > 0:
        minutes.append(a)
    else:
        excluded.append(i)


all_fc_data = fc_data.drop(excluded)
final_subs = all_fc_data.index


num = len(final_subs)
amount = []
for i in range(num):
    a = minutes[i]
    b = a.to_numpy()
    d = [float(c) for c in b]
    e = mean(d)
    amount.append(e)



fd = []
for i in final_subs:
    a = motion_df.loc[motion_df['subject'] == i, 'fd']
    fd.append(a)


avs = []
for i in range(num):
    a = fd[i]
    b = a.to_numpy()
    d = [float(c) for c in b]
    e = mean(d)
    avs.append(e)






data = {'subject' : final_subs, 'minutes' : amount, 'fd' : avs}

behav_data = pd.DataFrame(data)
behav_data.set_index('subject', inplace=True)



def mk_kfold_indices(subj_list, k = 10):
    """
    Splits list of subjects into k folds for cross-validation.
    """
    
    n_subs = len(subj_list)
    n_subs_per_fold = n_subs//k # floor integer for n_subs_per_fold

    indices = [[fold_no]*n_subs_per_fold for fold_no in range(k)] # generate repmat list of indices
    remainder = n_subs % k # figure out how many subs are left over
    remainder_inds = list(range(remainder))
    indices = [item for sublist in indices for item in sublist]    
    [indices.append(ind) for ind in remainder_inds] # add indices for remainder subs

    assert len(indices)==n_subs, "Length of indices list does not equal number of subjects, something went wrong"

    np.random.shuffle(indices) # shuffles in place

    return np.array(indices)


def split_train_test(subj_list, indices, test_fold):
    """
    For a subj list, k-fold indices, and given fold, returns lists of train_subs and test_subs
    """

    train_inds = np.where(indices!=test_fold)
    test_inds = np.where(indices==test_fold)

    train_subs = []
    for sub in subj_list[train_inds]:
        train_subs.append(sub)

    test_subs = []
    for sub in subj_list[test_inds]:
        test_subs.append(sub)

    return (train_subs, test_subs)



def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):

    """
    Extracts requested FC and behavioral data for a list of train_subs and test_subs
    """

    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]

    train_behav = behav_data.loc[train_subs, behav]
    train_motion = behav_data.loc[train_subs, 'fd']

    return (train_vcts, train_behav, test_vcts, train_motion)



def select_features(train_vcts, train_behav, train_motion, r_thresh=0.1, corr_type='pearson', verbose=False):
    
    """
    Runs the CPM feature selection step: 
    - correlates each edge with behavior, and returns a mask of edges that are correlated above some threshold, one for each tail (positive and negative)
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector
    if corr_type =='pearson':
        cov_ab = np.dot(train_behav.T - train_behav.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0]-1)
        corr_ab = cov_ab / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
        cov_ac = np.dot(train_behav.T - train_behav.mean(), train_motion.T - train_motion.mean()) / (train_behav.shape[0]-1)
        corr_ac = cov_ac / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_motion, ddof=1))
        cov_bc = np.dot(train_motion.T - train_motion.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0]-1)
        corr_bc = cov_bc / np.sqrt(np.var(train_motion, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
        cov_abc = corr_ab - (corr_ac * corr_bc)
        corr_abc = cov_abc / np.sqrt((1-(corr_ac**2)) * (1-(corr_bc**2)))

    elif corr_type =='spearman':
        corr = []
        for edge in train_vcts.columns:
            r_val = sp.stats.spearmanr(train_vcts.loc[:,edge], train_behav)[0]
            corr.append(r_val)

    # Define positive and negative masks
    mask_dict = {}
    mask_dict["pos"] = corr_abc > r_thresh
    mask_dict["neg"] = corr_abc < -r_thresh
    
    if verbose:
        print("Found ({}/{}) edges positively/negatively correlated with behavior in the training set".format(mask_dict["pos"].sum(), mask_dict["neg"].sum())) # for debugging

    return mask_dict



def build_model(train_vcts, mask_dict, train_behav):
    """
    Builds a CPM model:
    - takes a feature mask, sums all edges in the mask for each subject, and uses simple linear regression to relate summed network strength to behavior
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    model_dict = {}

    # Loop through pos and neg tails
    X_glm = np.zeros((train_vcts.shape[0], len(mask_dict.items())))

    t = 0
    for tail, mask in mask_dict.items():
        X = train_vcts.values[:, mask].sum(axis=1)
        X_glm[:, t] = X
        y = train_behav
        (slope, intercept) = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

    return model_dict



def apply_model(test_vcts, mask_dict, model_dict):
    """
    Applies a previously trained linear regression model to a test set to generate predictions of behavior.
    """

    behav_pred = {}

    X_glm = np.zeros((test_vcts.shape[0], len(mask_dict.items())))

    # Loop through pos and neg tails
    t = 0
    for tail, mask in mask_dict.items():
        X = test_vcts.loc[:, mask].sum(axis=1)
        X_glm[:, t] = X

        slope, intercept = model_dict[tail]
        behav_pred[tail] = slope*X + intercept
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    behav_pred["glm"] = np.dot(X_glm, model_dict["glm"])

    return behav_pred


def cpm_wrapper(all_fc_data, all_behav_data, behav, k=10, **cpm_kwargs):

    assert all_fc_data.index.equals(all_behav_data.index), "Row (subject) indices of FC vcts and behavior don't match!"

    subj_list = all_fc_data.index # get subj_list from df index
    
    indices = mk_kfold_indices(subj_list, k=k)
    
    # Initialize df for storing observed and predicted behavior
    col_list = []
    for tail in ["pos", "neg", "glm"]:
        col_list.append(behav + " predicted (" + tail + ")")
    col_list.append(behav + " observed")
    behav_obs_pred = pd.DataFrame(index=subj_list, columns = col_list)
    
    # Initialize array for storing feature masks
    n_edges = all_fc_data.shape[1]
    all_masks = {}
    all_masks["pos"] = np.zeros((k, n_edges))
    all_masks["neg"] = np.zeros((k, n_edges))
    
    for fold in range(k):
        print("doing fold {}".format(fold))
        train_subs, test_subs = split_train_test(subj_list, indices, test_fold=fold)
        train_vcts, train_behav, test_vcts, train_motion = get_train_test_data(all_fc_data, train_subs, test_subs, all_behav_data, behav=behav)
        mask_dict = select_features(train_vcts, train_behav, train_motion, **cpm_kwargs)
        all_masks["pos"][fold,:] = mask_dict["pos"]
        all_masks["neg"][fold,:] = mask_dict["neg"]
        model_dict = build_model(train_vcts, mask_dict, train_behav)
        behav_pred = apply_model(test_vcts, mask_dict, model_dict)
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, behav + " predicted (" + tail + ")"] = predictions
            
    behav_obs_pred.loc[subj_list, behav + " observed"] = all_behav_data[behav]
    
    return behav_obs_pred, all_masks



#sort data
behav_data.index.name = None 




#run cpm
cpm_kwargs = {'r_thresh': 0.1, 'corr_type': 'pearson'} 
behav_obs_pred, all_masks = cpm_wrapper(all_fc_data, behav_data, behav='minutes', **cpm_kwargs)





def plot_predictions(behav_obs_pred, tail="glm"):
    x = behav_obs_pred.filter(regex=("obs")).astype(float)
    x = x['minutes observed']
    y = behav_obs_pred.filter(regex=(tail)).astype(float)
    y = y['minutes predicted (glm)']
    hm = behav_data['fd']

    g = sea.regplot(x=x.T.squeeze(), y=y.T.squeeze(), color='gray')
    ax_min = min(min(g.get_xlim()), min(g.get_ylim()))
    ax_max = max(max(g.get_xlim()), max(g.get_ylim()))
    g.set_xlim(ax_min, ax_max)
    g.set_ylim(300, 500)
    g.set_box_aspect(1)
    
    data = {'x': x, 'y': y, 'hm': hm}
    df = pd.DataFrame(data)
    r = pg.partial_corr(data=df, x='x', y='y', covar='hm', method='pearson')
    r_val = r['r'].item()
    p_val = r['p-val'].item()
    subs = len(behav_obs_pred)
    g.annotate('r = {0:.2f}'.format(r_val), xy = (0.7, 0.2), xycoords = 'axes fraction')
    g.annotate('p = {0:.2f}'.format(p_val), xy = (0.7, 0.15), xycoords = 'axes fraction')
    g.annotate('n = {}'.format(subs), xy = (0.7, 0.1), xycoords = 'axes fraction')

    
    return g
    



#plot
condition = 'Rest'
g = plot_predictions(behav_obs_pred)
g.set_title(condition)
plt.show()
