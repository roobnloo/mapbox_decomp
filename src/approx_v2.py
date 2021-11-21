import time
from copy import deepcopy
from tensorly.random import random_cp
import numpy as np
from tensorly.decomposition import CP, parafac, non_negative_parafac, non_negative_parafac_hals
import tensorly as tl
from tensorly.decomposition._nn_cp import initialize_nn_cp
from tensorly.cp_tensor import CPTensor
from tensorly.metrics.regression import RMSE
import configparser
import matplotlib.pyplot as plt

def rank_approximator(path_to_tensor, rank_params, dir_to_save_factors, path_to_save_figures, testing=True):
    """Loops over a range of ranks for tensor approximation via PARAFAC decomposition. 
    
    Parameters:
    ----------- 
        path_to_tensor: 
        ranks_list: list with start, end, step arguments for np.arange() function
        dir_to_save_factors:
        path_to_save_figures:  
    
    Returns: 
    --------
        Saves factors and a figure to the specified dirs
    """
    # define random tensor for testing 
    if testing: 
        mapbox_tensor = random_cp((10, 10, 10), 3, random_state=1234, full=True)
    else:
        # read in original tensor 
        mapbox_tensor = tl.tensor(np.load(path_to_tensor))
    
    print(f"Tensor of size {mapbox_tensor.shape} imported")
    
    # initiate vectors 
    #ranks = np.arange(5, 50, 3)
    ranks = np.arange(rank_params[0], rank_params[1], rank_params[2])
    err = np.empty_like(ranks)
    err_ls = np.empty_like(ranks)
    tt = np.empty_like(ranks)
    tt_ls = np.empty_like(ranks)

    # run decomposition 
    # alternate decomposition rank
    for ii, r in enumerate(ranks):
        start = time.time()
        
        cp = CP(rank=r, tol=10e-3, linesearch=True, mask=~np.isnan(mapbox_tensor))
        fac = cp.fit_transform(np.nan_to_num(mapbox_tensor))
        
        tt[ii] = time.time() - start
        err[ii] = tl.norm(tl.cp_to_tensor(fac, mask=~np.isnan(mapbox_tensor)) - np.nan_to_num(mapbox_tensor))
        
        print('Execution time: ' + str("{:.2f}".format(tt[ii])) + ' ' + 'seconds')

        # Calculate the error of both decompositions
        # estimation error
        #est_err = tl.cp_to_tensor(fac_ls) - mapbox_tensor
        #err_ls[ii] = RMSE(np.nan_to_num(mapbox_tensor), fac_ls)

        (weights, factors) = fac # get the factors out to store them to drive
        # save factors to file 
        for i, fct in enumerate(factors):
            np.save(dir_to_save_factors + f"rank_{r}_factor_{i}", fct)
            np.save(dir_to_save_factors + f"rank_{r}_weights", weights)

    
    # plotting magic 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,6))

    #ax.loglog(tt, err - err_min, '.', label="No line search")
    #ax.loglog(tt_ls, err_ls, '.r', label="Line search")
    ax1.plot(ranks, err)
    #ax1.legend()
    ax1.set_ylabel("Error")
    ax1.set_xlabel("Ranks")

    ax2.scatter(ranks, tt)
    #ax2.legend()
    ax2.set_ylabel("Time")
    ax2.set_xlabel("Ranks")

    fig.savefig(path_to_save_figures)

    np.save(dir_to_save_factors + f"approximation_errors", err)
    np.save(dir_to_save_factors + f"approximation_ranks", ranks)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_tensor = config['PATHS']['path_to_tensor']
    path_to_save_figures = config['PATHS']['path_to_save_figures']
    dir_to_save_factors = config['PATHS']['dir_to_save_factors']
    rank_params = [5,50,2]
    rank_approximator(path_to_tensor, rank_params, dir_to_save_factors, path_to_save_figures, testing=False)
