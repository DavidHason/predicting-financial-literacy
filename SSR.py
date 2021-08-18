#!/usr/bin/env python
# coding: utf-8

# ### Prediction of Literacy Rate using Semi Supervised Learning basedRegression and SMOGN
# ---
# 
# 
# Author - David H.


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error

import seaborn as sns
from datetime import datetime
import itertools
import datetime
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


params = {'legend.fontsize': '20',
          'figure.figsize': (15, 5),
         'axes.labelsize': '18',
         'axes.titlesize':'30',
         'xtick.labelsize':'16',
         'ytick.labelsize':'16'}
plt.rcParams.update(params)

plt.rcParams['text.color'] = '#A04000'
plt.rcParams['xtick.color'] = '#800000'
plt.rcParams['ytick.color'] = '#808000'
plt.rcParams['axes.labelcolor'] = '#283747'


# In[2]:


df = pd.read_csv('data/CFS_2017-2018_FL.csv')
df.head()


# In[3]:


### Data Preprocessing


# In[4]:


df.drop(['new_id'], axis = 1, inplace=True)


# In[5]:


df['Financial Literacy'].plot(kind = 'line', figsize = (17,6));


# ### Pre-processing Data

# In[6]:


df.info()


# In[7]:


from sklearn import preprocessing

for col in df.select_dtypes(include=['object']).columns:
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'species'.
    df[col]= label_encoder.fit_transform(df[col].astype(str))


# In[8]:


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as r2
import smogn
## load dependencies - internal
from smogn.phi import phi
from smogn.phi_ctrl_pts import phi_ctrl_pts
from smogn.over_sampling import over_sampling

## synthetic minority over-sampling technique for regression with gaussian noise 
def smoter_smogn(
    
    ## main arguments / inputs
    data,                     ## training set (pandas dataframe)
    y,                        ## response variable y by name (string)
    k = 5,                    ## num of neighs for over-sampling (pos int)
    pert = 0.02,              ## perturbation / noise percentage (pos real)
    samp_method = "balance",  ## over / under sampling ("balance" or extreme")
    under_samp = True,        ## under sampling (bool)
    drop_na_col = True,       ## auto drop columns with nan's (bool)
    drop_na_row = True,       ## auto drop rows with nan's (bool)
    replace = False,          ## sampling replacement (bool)
    
    ## phi relevance function arguments / inputs
    rel_thres = 0.2,          ## relevance threshold considered rare (pos real)
    rel_method = "auto",      ## relevance method ("auto" or "manual")
    rel_xtrm_type = "both",   ## distribution focus ("high", "low", "both")
    rel_coef = 1.5,           ## coefficient for box plot (pos real)
    rel_ctrl_pts_rg = None    ## input for "manual" rel method  (2d array)
    
    ):
    
    """
    the main function, designed to help solve the problem of imbalanced data 
    for regression, much the same as SMOTE for classification; SMOGN applies 
    the combintation of under-sampling the majority class (in the case of 
    regression, values commonly found near the mean of a normal distribution 
    in the response variable y) and over-sampling the minority class (rare 
    values in a normal distribution of y, typically found at the tails)
    
    procedure begins with a series of pre-processing steps, and to ensure no 
    missing values (nan's), sorts the values in the response variable y by
    ascending order, and fits a function 'phi' to y, corresponding phi values 
    (between 0 and 1) are generated for each value in y, the phi values are 
    then used to determine if an observation is either normal or rare by the 
    threshold specified in the argument 'rel_thres' 
    
    normal observations are placed into a majority class subset (normal bin) 
    and are under-sampled, while rare observations are placed in a seperate 
    minority class subset (rare bin) where they're over-sampled
    
    under-sampling is applied by a random sampling from the normal bin based 
    on a calculated percentage control by the argument 'samp_method', if the 
    specified input of 'samp_method' is "balance", less under-sampling (and 
    over-sampling) is conducted, and if "extreme" is specified more under-
    sampling (and over-sampling is conducted)
    
    over-sampling is applied one of two ways, either synthetic minority over-
    sampling technique for regression 'smoter' or 'smoter-gn' which applies a 
    similar interpolation method to 'smoter', but takes an additional step to
    perturb the interpolated values with gaussian noise
    
    'smoter' is selected when the distance between a given observation and a 
    selected nearest neighbor is within the maximum threshold (half the median 
    distance of k nearest neighbors) 'smoter-gn' is selected when a given 
    observation and a selected nearest neighbor exceeds that same threshold
    
    both 'smoter' and 'smoter-gn' are only applied to numeric / continuous 
    features, synthetic values found in nominal / categorical features, are 
    generated by randomly selecting observed values found within their 
    respective feature
    
    procedure concludes by post-processing and returns a modified pandas data
    frame containing under-sampled and over-sampled (synthetic) observations, 
    the distribution of the response variable y should more appropriately 
    reflect the minority class areas of interest in y that are under-
    represented in the original training set
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    """
    
    ## pre-process missing values
    if bool(drop_na_col) == True:
        data = data.dropna(axis = 1)  ## drop columns with nan's
    
    if bool(drop_na_row) == True:
        data = data.dropna(axis = 0)  ## drop rows with nan's
    
    ## quality check for missing values in dataframe
    if data.isnull().values.any():
        raise ValueError("cannot proceed: data cannot contain NaN values")
    
    ## quality check for y
    if isinstance(y, str) is False:
        raise ValueError("cannot proceed: y must be a string")
    
    if y in data.columns.values is False:
        raise ValueError("cannot proceed: y must be an header name (string)                found in the dataframe")
    
    ## quality check for k number specification
    if k > len(data):
        raise ValueError("cannot proceed: k is greater than number of                observations / rows contained in the dataframe")
    
    ## quality check for perturbation
    if pert > 1 or pert <= 0:
        raise ValueError("pert must be a real number number: 0 < R < 1")
    
    ## quality check for sampling method
    if samp_method in ["balance", "extreme"] is False:
        raise ValueError("samp_method must be either: 'balance' or 'extreme' ")
    
    ## quality check for relevance threshold parameter
    if rel_thres == None:
        raise ValueError("cannot proceed: relevance threshold required")
    
    if rel_thres > 1 or rel_thres <= 0:
        raise ValueError("rel_thres must be a real number number: 0 < R < 1")
    
    ## store data dimensions
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## determine column position for response variable y
    y_col = data.columns.get_loc(y)
    
    ## move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]
    
    ## store original feature headers and
    ## encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)
    
    ## sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by = d - 1)
    y_sort = y_sort[d - 1]
    
    ## -------------------------------- phi --------------------------------- ##
    ## calculate parameters for phi relevance function
    ## (see 'phi_ctrl_pts()' function for details)
    phi_params = phi_ctrl_pts(
        
        y = y_sort,                ## y (ascending)
        method = rel_method,       ## defaults "auto" 
        xtrm_type = rel_xtrm_type, ## defaults "both"
        coef = rel_coef,           ## defaults 1.5
        ctrl_pts = rel_ctrl_pts_rg ## user spec
    )
    
    ## calculate the phi relevance function
    ## (see 'phi()' function for details)
    y_phi = phi(
        
        y = y_sort,                ## y (ascending)
        ctrl_pts = phi_params      ## from 'phi_ctrl_pts()'
    )
    
    ## phi relevance quality check
    if all(i == 0 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 1")
    
    if all(i == 1 for i in y_phi):
        raise ValueError("redefine phi relevance function: all points are 0")
    ## ---------------------------------------------------------------------- ##
    
    ## determine bin (rare or normal) by bump classification
    bumps = [0]
    
    for i in range(0, len(y_sort) - 1):
        if ((y_phi[i] >= rel_thres and y_phi[i + 1] < rel_thres) or 
            (y_phi[i] < rel_thres and y_phi[i + 1] >= rel_thres)):
                bumps.append(i + 1)
    
    bumps.append(n)
    
    ## number of bump classes
    n_bumps = len(bumps) - 1
    
    ## determine indicies for each bump classification
    b_index = {}
    
    for i in range(n_bumps):
        b_index.update({i: y_sort[bumps[i]:bumps[i + 1]]})
    
    ## calculate over / under sampling percentage according to
    ## bump class and user specified method ("balance" or "extreme")
    b = round(n / n_bumps)
    s_perc = []
    scale = []
    obj = []
    
    if samp_method == "balance":
        for i in b_index:
            s_perc.append(b / len(b_index[i]))
            
    if samp_method == "extreme":
        for i in b_index:
            scale.append(b ** 2 / len(b_index[i]))
        scale = n_bumps * b / sum(scale)
        
        for i in b_index:
            obj.append(round(b ** 2 / len(b_index[i]) * scale, 2))
            s_perc.append(round(obj[i] / len(b_index[i]), 1))
    
    ## conduct over / under sampling and store modified training set
    data_new = pd.DataFrame()
    
    for i in range(n_bumps):
        
        ## no sampling
        if s_perc[i] == 1:
            
            ## simply return no sampling
            ## results to modified training set
            data_new = pd.concat([data.iloc[b_index[i].index], data_new])
        
        ## over-sampling
        if s_perc[i] > 1:
            
            ## generate synthetic observations in training set
            ## considered 'minority'
            ## (see 'over_sampling()' function for details)
            synth_obs = over_sampling(
                data = data,
                index = list(b_index[i].index),
                perc = s_perc[i],
                pert = pert,
                k = k
            )
            
            ## concatenate over-sampling
            ## results to modified training set
            data_new = pd.concat([synth_obs, data_new])
        
        ## under-sampling
        if under_samp is True:
            if s_perc[i] < 1:
                
                ## drop observations in training set
                ## considered 'normal' (not 'rare')
                omit_index = np.random.choice(
                    a = list(b_index[i].index), 
                    size = int(s_perc[i] * len(b_index[i])),
                    replace = replace
                )
                
                omit_obs = data.drop(
                    index = omit_index, 
                    axis = 0
                )
                
                ## concatenate under-sampling
                ## results to modified training set
                data_new = pd.concat([omit_obs, data_new])
    
    ## rename feature headers to originals
    data_new.columns = feat_names
    
    ## restore response variable y to original position
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data_new = data_new[data_new.columns[cols]]
    
    ## restore original data types
    for j in range(d):
        data_new.iloc[:, j] = data_new.iloc[:, j].astype(feat_dtypes_orig[j])
    
    ## return modified training set
    return data_new
def squaredr(y_true, y_pred):
    return (-r2(y_true, y_pred)*0.01)**2

def load_data(data_dir):
    """
    Loads data from data directory.
    """
    X = np.load(os.path.join(data_dir, 'X.npy'), allow_pickle=True)
    y = np.load(os.path.join(data_dir, 'y.npy'),allow_pickle=True)
    return X, y.reshape(-1, 1)


# In[9]:


df['Financial Literacy'].value_counts()


# In[10]:


rg_mtrx_1 = [
    #[.750,  0, 0],  ## under-sample ("minority")
    [.625,  0, 0],  ## under-sample ("minority")
    [.875,  0, 0],  ## under-sample ("minority")
    [0.500,  0, 0],  ## under-sample ("minority")

    [.000,  1, 0],  ## over-sample ("minority")
    [.125,  1, 0],  ## over-sample ("minority")
    [.250,  1, 0],  ## over-sample ("minority")
    #[1.0,  1, 0],  ## over-sample ("minority")
    [.375,  1, 0],  ## over-sample ("minority")
]
df_smogn = smoter_smogn(df,y='Financial Literacy',k=5,samp_method='balance',pert=0.05,rel_thres = 0.25,rel_method = 'manual',rel_ctrl_pts_rg = rg_mtrx_1)


# In[15]:


# Generate data on commute times.
plt.figure(figsize = (17,6))
df['Financial Literacy'].plot.hist(grid=True, bins=8, rwidth=0.9,
                   color='#607c8e')
plt.title('Distribution of target Variable before applying SMOGN')
plt.xlabel("Value");
plt.ylabel("Counts");
plt.grid(axis='y', alpha=0.75)
plt.show();


# In[14]:


# Generate data on commute times.
plt.figure(figsize = (17,6))
df_smogn['Financial Literacy'].plot.hist(grid=True, bins=8, rwidth=0.9,
                   color='#607c8e')
plt.title('Distribution of target Variable after applying SMOGN')
plt.xlabel("Value");
plt.ylabel("Counts");
plt.grid(axis='y', alpha=0.75)
plt.show();


# ## SSR

# In[16]:


import pandas as pd
import numpy as np
from time import time
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# In[17]:


#X = df.drop(['Financial Literacy'], axis =1)
#y = df['Financial Literacy']

X = df_smogn.iloc[:,1:]
y = df_smogn.iloc[:,0]
# Save to numpy array
np.save('datasets/X.npy', np.array(X))
np.save('datasets/y.npy', np.array(y))


# ## COREG
# 

# In[18]:


from time import time
import os

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor

def load_data(data_dir):
    """
    Loads data from data directory.
    """
    print("loading the data ...")
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    print("Performing feature selection ...")

    estimator = ExtraTreesRegressor(n_estimators=100, random_state=0)
    selector = RFE(estimator, n_features_to_select = 30, step=1)
    selector = selector.fit(X[~np.isnan(y)], y[~np.isnan(y)])
    X = X[: , selector.support_]
    return X, y.reshape(-1, 1)


class Coreg():
    """
    Instantiates a CoReg regressor.
    """
    def __init__(self, k1=3, k2=3, p1=2, p2=5, max_iters=100, pool_size=100):
        self.k1, self.k2 = k1, k2  # number of neighbors
        self.p1, self.p2 = p1, p2  # distance metrics
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2 = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.h1_temp = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2_temp = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)

    def add_data(self, data_dir):
        """
        Adds data and splits into labeled and unlabeled.
        """
        self.X, self.y = load_data(data_dir)

    def run_trials(self, num_train=100, trials=10, verbose=True):
        """
        Runs multiple trials of training.
        """
        self.num_train = num_train
        self.num_trials = trials
        self._initialize_metrics()
        self.trial = 0
        while self.trial < self.num_trials:
            t0 = time()
            print('Starting trial {}:'.format(self.trial + 1))
            self.train(
                random_state=(self.trial+self.num_train),
                num_labeled=self.num_train, num_test=100, verbose=verbose)
            print('Finished trial {}: {:0.2f}s elapsed\n'.format(
                self.trial + 1, time() - t0))
            self.trial += 1

    def train(self, random_state=-1, num_labeled=100, num_test=100,
              verbose=False, store_results=False):

        print("Trains the CoReg regressor")
        t0 = time()
        self._split_data(random_state, num_labeled, num_test)
        self._fit_and_evaluate(verbose)
        self._get_pool()
        if verbose:
            print('Initialized h1, h2: {:0.2f}s\n'.format(time()-t0))
        for t in range(1, self.max_iters+1):
            stop_training = self._run_iteration(t, t0, verbose, store_results)
            if stop_training:
                if verbose:
                    print('Done in {} iterations: {:0.2f}s'.format(t, time()-t0))
                break
        if verbose:
            print('Finished {} iterations: {:0.2f}s'.format(t, time()-t0))

    def _run_iteration(self, t, t0, verbose=False, store_results=False):
        """
        Run t-th iteration of co-training, returns stop_training=True if
        no more unlabeled points are added to label sets.
        """
        stop_training = False
        if verbose:
            print('Started iteration {}: {:0.2f}s'.format(t, time()-t0))
        self._find_points_to_add()
        added = self._add_points()
        if added:
            self._fit_and_evaluate(verbose)
            if store_results:
                self._store_results(t)
            self._remove_from_unlabeled()
            self._get_pool()
        else:
            stop_training = True
        return stop_training

    def _add_points(self):
        """
        Adds new examples to training sets.
        """
        added = False
        if self.to_add['x1'] is not None:
            self.L2_X = np.vstack((self.L2_X, self.to_add['x1']))
            self.L2_y = np.vstack((self.L2_y, self.to_add['y1']))
            added = True
        if self.to_add['x2'] is not None:
            self.L1_X = np.vstack((self.L1_X, self.to_add['x2']))
            self.L1_y = np.vstack((self.L1_y, self.to_add['y2']))
            added = True
        return added

    def _compute_delta(self, omega, L_X, L_y, h, h_temp):
        """
        Computes the improvement in MSE among the neighbors of the point being
        evaluated.
        """
        delta = 0
        for idx_o in omega:
            delta += (L_y[idx_o].reshape(1, -1) -
                      h.predict(L_X[idx_o].reshape(1, -1))) ** 2
            delta -= (L_y[idx_o].reshape(1, -1) -
                      h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2
        return delta

    def _compute_deltas(self, L_X, L_y, h, h_temp):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        deltas = np.zeros((self.U_X_pool.shape[0],))
        for idx_u, x_u in enumerate(self.U_X_pool):
            # Make prediction
            x_u = x_u.reshape(1, -1)
            y_u_hat = h.predict(x_u).reshape(1, -1)
            # Compute neighbors
            omega = h.kneighbors(x_u, return_distance=False)[0]
            # Retrain regressor after adding unlabeled point
            X_temp = np.vstack((L_X, x_u))
            y_temp = np.vstack((L_y, y_u_hat)) # use predicted y_u_hat
            h_temp.fit(X_temp, y_temp)
            delta = self._compute_delta(omega, L_X, L_y, h, h_temp)
            deltas[idx_u] = delta
        return deltas

    def _evaluate_metrics(self, verbose):
        """
        Evaluates KNN regressors on training and test data.
        """
        test1_hat = self.h1.predict(self.X_labeled)
        test2_hat = self.h2.predict(self.X_labeled)
        test_hat = 0.5 * (test1_hat + test2_hat)
        self.mse1 = mean_squared_error(test1_hat, self.y_labeled)
        self.mse2 = mean_squared_error(test2_hat, self.y_labeled)
        self.mse = mean_squared_error(test_hat, self.y_labeled)
        self.r2score = r2_score(self.y_labeled, test_hat)
        self.rmse = np.sqrt(mean_squared_error(test_hat, self.y_labeled))
        self.mae = mean_absolute_error(test_hat, self.y_labeled)
        results = []
        if verbose:
            print('MSEs:')
            print('  KNN1:')
            print('    Test: {:0.4f}'.format(self.mse1))
            print('  KNN2:')
            print('    Test: {:0.4f}'.format(self.mse2))
            print('  Combined:')
            print('  MSE:')
            print('    Test: {:0.4f}'.format(self.mse))
            print('  R-Squared:')
            print('    Test: {:0.4f}'.format(self.r2score))
            print('  RMSE:')
            print('    Test: {:0.4f}'.format(self.rmse))
            print('   MAE')
            print('    Test: {:0.4f}'.format(self.mae))
        results.append(self.mse)    
        results.append(self.r2score)    
        results.append(self.rmse)    
        results.append(self.mae)    

        return results
        
    def _find_points_to_add(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                       'x2': None, 'y2': None, 'idx2': None}
        # Keep track of added idxs
        added_idxs = []
        for idx_h in [1, 2]:
            if idx_h == 1:
                h = self.h1
                h_temp = self.h1_temp
                L_X, L_y = self.L1_X, self.L1_y
            elif idx_h == 2:
                h = self.h2
                h_temp = self.h2_temp
                L_X, L_y = self.L2_X, self.L2_y
            deltas = self._compute_deltas(L_X, L_y, h, h_temp)
            
    def _fit_and_evaluate(self, verbose):
        """
        Fits h1 and h2 and evaluates metrics.
        """
        self.h1.fit(self.L1_X, self.L1_y)
        self.h2.fit(self.L2_X, self.L2_y)
        self._evaluate_metrics(verbose)

    def _get_pool(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U_X_pool, self.U_y_pool, self.U_idx_pool = shuffle(
            self.U_X, self.U_y, range(self.U_y.size))
        self.U_X_pool = self.U_X_pool[:self.pool_size]
        self.U_y_pool = self.U_y_pool[:self.pool_size]
        self.U_idx_pool = self.U_idx_pool[:self.pool_size]

    def _initialize_metrics(self):
        """
        Sets up metrics to be stored.
        """
        initial_metrics = np.full((self.num_trials, self.max_iters+1), np.inf)
        self.mses1_train = np.copy(initial_metrics)
        self.mses1_test = np.copy(initial_metrics)
        self.mses2_train = np.copy(initial_metrics)
        self.mses2_test = np.copy(initial_metrics)
        self.mses_train = np.copy(initial_metrics)
        self.mses_test = np.copy(initial_metrics)

    def _remove_from_unlabeled(self):
        # Remove added examples from unlabeled
        to_remove = []
        if self.to_add['idx1'] is not None:
            to_remove.append(self.to_add['idx1'])
        if self.to_add['idx2'] is not None:
            to_remove.append(self.to_add['idx2'])
        self.U_X = np.delete(self.U_X, to_remove, axis=0)
        self.U_y = np.delete(self.U_y, to_remove, axis=0)

    def _split_data(self, random_state=-1, num_labeled=100, num_test=600):
        """
        Shuffles data and splits it into train, test, and unlabeled sets.
        """
        if random_state >= 0:
            self.X_shuffled, self.y_shuffled, self.shuffled_indices = shuffle(
                self.X, self.y, range(self.y.size), random_state=random_state)
        else:
            self.X_shuffled = self.X[:]
            self.y_shuffled = self.y[:]
            self.shuffled_indices = range(self.y.size)
        # Initial labeled, test, and unlabeled sets
        '''test_end = num_labeled + num_test
        self.X_labeled = self.X_shuffled[:num_labeled]
        self.y_labeled = self.y_shuffled[:num_labeled]
        self.X_test = self.X_shuffled[num_labeled:test_end]
        self.y_test = self.y_shuffled[num_labeled:test_end]
        self.X_unlabeled = self.X_shuffled[test_end:]
        self.y_unlabeled = self.y_shuffled[test_end:]
        # Up-to-date training sets and unlabeled set
        self.L1_X = self.X_labeled[:]
        self.L1_y = self.y_labeled[:]
        self.L2_X = self.X_labeled[:]
        self.L2_y = self.y_labeled[:]
        self.U_X = self.X_unlabeled[:]
        self.U_y = self.y_unlabeled[:]'''
        ########### split the data into labeled and unlabeled based on missing entries in the y
        not_missing_indices = np.concatenate(~np.isnan(self.y_shuffled)).ravel()
        missing_indices = np.concatenate(np.isnan(self.y_shuffled)).ravel()

        self.X_labeled = self.X_shuffled[not_missing_indices,:]
        self.y_labeled = self.y_shuffled[not_missing_indices,:]
        self.X_test = self.X_labeled[-num_test:,:]
        self.y_test = self.y_labeled[-num_test:,:]
        self.X_labeled = self.X_labeled[:-num_test,:]
        self.y_labeled = self.y_labeled[:-num_test,:]
        self.X_unlabeled = self.X_shuffled[missing_indices,:]
        self.y_unlabeled = self.y_shuffled[missing_indices,:]
        # Up-to-date training sets and unlabeled set
        self.L1_X = self.X_labeled[:]
        self.L1_y = self.y_labeled[:]
        self.L2_X = self.X_labeled[:]
        self.L2_y = self.y_labeled[:]
        self.U_X = self.X_unlabeled[:]
        self.U_y = self.y_unlabeled[:]


# In[19]:


get_ipython().run_cell_magic('time', '', "\nstore_results = False\nresults = []\ndata_dir = 'datasets'\nk1 = 2\nk2 = 2\np1 = 2\np2 = 2\nmax_iters = 100\npool_size = 100\nverbose = True\nrandom_state = -1\nnum_labeled = 500\nnum_test = 100\n\ncr = Coreg(k1, k2, p1, p2, max_iters, pool_size)\ncr.add_data(data_dir)\n\n# Run training\nnum_train = 500\ntrials = 100\nverbose = True\n\ncr.run_trials(num_train, trials, verbose)")


# In[105]:


get_ipython().run_cell_magic('time', '', '\n\ndata_dir = \'datasets/\'\nk1 = 2\nk2 = 2\np1 = 2\np2 = 3\nmax_iters = 500\npool_size = 100\nverbose = True\nrandom_state = -1\nnum_labeled = 500\nnum_test = 100\n\ncr = Coreg(k1, k2, p1, p2, max_iters, pool_size)\n\n###################################################\n\ncr.add_data(data_dir)\n\n####\n\ncr._split_data(random_state, num_labeled, num_test)\n\ndef test_split_data():\n    assert np.allclose(cr.L1_X, cr.X_labeled)\n\n####\n\n## Evaluate\n\nprint("==============")\nprint("Evaluate CoReg")\nprint("==============")\n\ncr._fit_and_evaluate(verbose=True)\ncr._get_pool()\n\nresults = cr._evaluate_metrics(verbose=False)\n\ndef test_iteration():\n    stop_training = cr._run_iteration(0, 0, False, False)\n    assert stop_training in [True, False]')


# In[106]:


metrics = ['MSE', 'R-Squared', 'RMSE', 'MAE'] 

plt.figure(figsize = (17,8))
plt.bar(metrics, results)
plt.title('Comparison of Metrics for SMOGN-COREG (SSR)')
plt.show();


# In[107]:


pd.DataFrame({'Metric' : metrics, 'Value' : results})


# # Results for different metrics for R

# In[50]:


indx = []
num_labeled = 500
num_test_list = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 495]
for numtest in num_test:
    indx.append(numtest*100/num_labeled)


# In[57]:


get_ipython().run_cell_magic('time', '', "data_dir = 'datasets/'\nk1 = 2\nk2 = 2\np1 = 2\np2 = 2\nmax_iters = 500\npool_size = 100\nverbose = True\nrandom_state = -1\n\nnum_labeled = 500\nnum_test_list = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 495]\n\nall_results = []\n\n\nfor numtest in num_test:\n    num_labeled = 500\n    num_test = numtest\n    cr = Coreg(k1, k2, p1, p2, max_iters, pool_size)\n    cr.add_data(data_dir)\n    cr._split_data(random_state, num_labeled, num_test)\n    cr._fit_and_evaluate(verbose=True)\n    cr._get_pool()\n\n    results = cr._evaluate_metrics(verbose)\n    all_results.append(results)")


# In[58]:


allresultsDF = pd.DataFrame(all_results, columns=metrics)
allresultsDF


# In[59]:


allresultsDF.index = indx
allresultsDF


# In[94]:


fig, ax1 = plt.subplots(figsize = (17,9))
plt.title('R-Squared & RMSE comparison with Labelled Ratio (R) %')
ax2 = ax1.twinx()
ax1.plot(allresultsDF.index, allresultsDF['R-Squared'], 'g-', label = 'R-Squared')
ax2.plot(allresultsDF.index, allresultsDF['RMSE'], 'b-', label="RMSE")

ax1.set_xlabel('Labelled Ratio (R) %')
ax1.set_ylabel('R-Squared', color='g')
ax1.legend(loc = 'upper left')
ax2.set_ylabel('RMSE', color='b')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show();


# In[108]:


fig, ax1 = plt.subplots(figsize = (17,9))
plt.title('R-Squared & RMSE comparison with Labelled Ratio (R) %')
ax2 = ax1.twinx()
ax1.plot(allresultsDF.index, allresultsDF['MAE'], 'g-', label = 'MAE')
ax2.plot(allresultsDF.index, allresultsDF['MSE'], 'b-', label="MSE")

ax1.set_xlabel('Labelled Ratio (R) %')
ax1.set_ylabel('R-Squared', color='g')
ax1.legend(loc = 'upper left')
ax2.set_ylabel('RMSE', color='b')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show();

