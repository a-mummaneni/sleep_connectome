
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
iterations = 1000
par_runs = int(iterations/size)


import os
import glob
import pandas as pd
import numpy as np
import pingouin as pg


#authored by Anu Mummaneni, adapted from es finn 


#functions to build model
#takes in inputs of equally sized and indexed fc matrices (train_vcts, expects subject x edges) and behavioral data (train_behav)

def select_features(train_vcts, train_behav, train_motion, behav_gender, behav_age, behav_runs, r_thresh=0.1, corr_type='pearson', verbose=False):
    
    """
    Runs the CPM feature selection step:
    - correlates each edge with behavior, and returns a mask of edges that are correlated above some threshold, one for each tail (positive and negative)
    """

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector

    corr_sp = []
       
    for edge in train_vcts.columns:
        x = train_vcts.loc[:,edge]
        y = train_behav['PSQI_AmtSleep']
        hm = train_motion['fd_mean']
        sex = behav_gender
        run = behav_runs
        age = behav_age
        data = {'x': x, 'y': y, 'hm': hm, 'sex' : sex, 'run' : run, 'age' : age}
        df = pd.DataFrame(data)
        r = pg.partial_corr(data=df, x='x', y='y', covar=['hm', 'sex', 'run', 'age'], method='spearman')
        r_val = r['r'].item()

            #r_val = sp.stats.spearmanr(train_vcts.loc[:,edge], train_behav)[0]
        corr_sp.append(r_val)
    

    # Define positive and negative masks
    mask_dict = {}
    
    corr_sp = np.array(corr_sp)

    mask_dict["pos"] = corr_sp > r_thresh
    mask_dict["neg"] = corr_sp < -r_thresh
    

  
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
        y = train_behav['PSQI_AmtSleep']
        (slope, intercept) = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

    return model_dict



def cpm_wrapper_train(train_vcts, train_behav, behav, k=10, **cpm_kwargs):

    assert train_vcts.index.equals(train_behav.index), "Row (subject) indices of FC vcts and behavior don't match!"
    
    # Initialize array for storing feature masks
    
    mask_dict = select_features(train_vcts, train_behav, train_motion, behav_gender, behav_age, behav_runs, **cpm_kwargs)
    model_dict = build_model(train_vcts, mask_dict, train_behav)
    
    return mask_dict, model_dict


test_vcts = test_vcts.sort_index(axis=0, ascending=True)
train_behav.index.name = None
test_behav.index.name = None

#these functions produce masks and model weights to be applied to a novel dataset
cpm_kwargs = {'r_thresh': 0.1, 'corr_type': 'spearman'}
mask_dict, model_dict = cpm_wrapper_train(train_vcts, train_behav, behav= 'PSQI_AmtSleep', **cpm_kwargs)







#functions to apply model
#takes in mask and model dict generated above
#inputs are similar to those listed above: fc matrices, formatted in a dataframe as subjects x edges, and a dataframe of equal length containing behavioral data

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



def cpm_wrapper_test(test_vcts, test_behav, behav, k=10, **cpm_kwargs):

    assert test_vcts.index.equals(test_behav.index), "Row (subject) indices of FC vcts and behavior don't match!"
    subj_list = test_behav.index
    
    # Initialize df for storing observed and predicted behavior
    col_list = []
    for tail in ["pos", "neg", "glm"]:
        col_list.append(behav + " predicted (" + tail + ")")
    col_list.append(behav + " observed")
    behav_obs_pred = pd.DataFrame(index=subj_list, columns = col_list)
    
    
    # Initialize array for storing feature masks
    

    behav_pred = apply_model(test_vcts, mask_dict, model_dict)
    for tail, predictions in behav_pred.items():
        behav_obs_pred.loc[subj_list, behav + " predicted (" + tail + ")"] = predictions
            
    behav_obs_pred.loc[subj_list, behav + " observed"] = test_behav[behav]
    
    return behav_obs_pred

#run final cpm
cpm_kwargs = {'r_thresh': 0.1, 'corr_type': 'spearman'}
behav_obs_pred = cpm_wrapper_test(test_vcts, test_behav, behav='minutes', **cpm_kwargs)




#function to run permutation testing in order

task = 'minutes'

def r_stats(behav_obs_pred, tail="glm"):
    x = behav_obs_pred.filter(regex=("obs")).astype(float)
    x = x['{} observed'.format(task)]
    y = behav_obs_pred.filter(regex=(tail)).astype(float)
    y = y['{} predicted (glm)'.format(task)]
    hm = behav_data['fd']
    sex = behav_data['sex']
    age = behav_data['age']

    data = {'x':x, 'y':y, 'hm':hm, 'sex':sex, 'age':age}
    df = pd.DataFrame(data)
    
    r = pg.partial_corr(data=df, x='x', y='y', covar=['hm','sex','age'], method='spearman')
    r_val = r['r'].item()
    print(r_val)
    
    p_val = r['p-val'].item()
    return r_val


#to perform correctly ordered permutations

#for across dataset, one 
hist = []
    
for i in range(1):
    print("rank ", rank, " working on iteration ", i)
    mask_dict, model_dict = cpm_wrapper_train(train_vcts, train_behav, behav='PSQI_AmtSleep', **cpm_kwargs)
    behav_obs_pred = cpm_wrapper_test(test_vcts, test_behav, behav='minutes', **cpm_kwargs)
    r = r_stats(behav_obs_pred)
    hist.append(r)
    print("rank ", rank, "done with iteration ", i)



#to perform null permutations
null_hist = []

subject = behav_data.index

for i in range(par_runs):
    print("rank ", rank, " working on iteration ", i)
    np.seterr(divide='ignore', invalid='ignore')
    bd = behav_data.sample(frac=1)
    bd['Subject'] = subject
    bd.set_index('Subject', inplace=True)
    mask_dict, model_dict = cpm_wrapper_train(train_vcts, train_behav, behav= 'PSQI_AmtSleep', **cpm_kwargs)
    behav_obs_pred = cpm_wrapper_test(test_vcts, b, behav='minutes', **cpm_kwargs)
    r = r_stats(behav_obs_pred)
    r = r_stats(behav_obs_pred)
    null_hist.append(r)
    print("rank ", rank, "done with iteration ", i)
    
#to obtain r and p values from permutation testing
fisher_hist = np.arctanh(hist)
fisher_mean = fisher_hist.mean()
mean = np.tanh(fisher_mean)
mean = hist.mean()

greater = null_hist[(null_hist[0] ==   xx #mean value
) | (null_hist[0] >   xx
)]
    
p_val = len(greater) / 1001
print(p_val)

#plot consistent edges

#read in shen labels
shen268_coords = pd.read_csv("shen268_coords.csv", index_col="NodeNo")
print(shen268_coords.shape)
shen268_coords.head()


#create readable dicts from mask_dicts, produced by cpm_wrapper_train function
n_edges = train_vcts.shape[1]
all_masks = {}
all_masks["pos"] = np.zeros((1, n_edges))
all_masks["neg"] = np.zeros((1, n_edges))

all_masks["pos"][0,:] = mask_dict["pos"]
all_masks["neg"][0,:] = mask_dict["neg"]

all_masks2["pos"] = np.zeros((1, n_edges))
all_masks2["neg"] = np.zeros((1, n_edges))

all_masks2["pos"][0,:] = mask_dict2["pos"]
all_masks2["neg"][0,:] = mask_dict2["neg"]


#this function takes in two dicts, each producced by the above code
def plot_consistent_edges(all_masks, all_masks2, tail, thresh =.8, color='gray'):

    edge_frac = (all_masks[tail].sum(axis=0))/(all_masks[tail].shape[0])
    edge_frac2 = (all_masks2[tail].sum(axis=0))/(all_masks2[tail].shape[0])
    data = {'Edge Frac 1' : edge_frac, 'Edge Frac 2' : edge_frac2}
    df = pd.DataFrame(data)
        
    df.loc[df['Edge Frac 1'] < thresh, 'Edge Frac 1'] = 0
    df.loc[df['Edge Frac 2'] < thresh, 'Edge Frac 1'] = 0
    df.loc[df['Edge Frac 1'] > thresh, 'Edge Frac 1'] = 1
    edge_frac_new = df['Edge Frac 1'].to_numpy()
    print("For the {} tail, {} edges were selected in at least {}% of folds".format(tail, (edge_frac_new>=thresh).sum(), thresh*100))

    edge_frac_square = sp.spatial.distance.squareform(edge_frac_new)
    nodes = edge_frac_square >= thresh

    node_mask = np.amax(edge_frac_square, axis=0) >= thresh # find nodes that have at least one edge that passes the threshold
    node_size = edge_frac_square.sum(axis=0)*node_mask*20 # size nodes based on how many suprathreshold edges they have
    plot_connectome(adjacency_matrix=edge_frac_square, edge_threshold=thresh,
                        node_color = color,
                        node_coords=shen268_coords, node_size=node_size,
                        display_mode= 'lzry',
                        edge_kwargs={"linewidth": 1, 'color': color, 'alpha' : 0.5})
        
    return edge_frac_square


pos = plot_consistent_edges(all_masks, all_masks2, "pos", thresh = .8, color = 'blue')
neg = plot_consistent_edges(all_masks, all_masks2, "neg", thresh = .8, color = 'red')



