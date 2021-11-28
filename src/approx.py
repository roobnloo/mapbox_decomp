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
import matplotlib.ticker as mticker
import csv
import gc

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

    # frob norm of centered tensor (TSS)
    x_c = tl.norm(np.nan_to_num(mapbox_tensor) - np.mean(np.nan_to_num(mapbox_tensor)))
    print(f"Estimated x_c: {x_c}")
    
    # initiate vectors 
    #ranks = np.arange(5, 50, 3)
    ranks = np.arange(rank_params[0], rank_params[1], rank_params[2])
    err_ls = np.zeros_like(ranks, dtype=float)
    tt_ls = np.zeros_like(ranks, dtype=float)
    var_ls = np.zeros_like(ranks, dtype=float)

    # run decomposition 
        # alternate decomposition rank
    for ii, r in enumerate(ranks):
        print(f"Estimating for rank {r}")
        #start = time()
        #cp = CP(rank=r, tol=10e-3, linesearch=True)
        #fac_ls = cp.fit_transform(mapbox_tensor)
        # we can also produce a good initial guess for NNCP via initialize_cp()
        tic = time.time()
        
        # decomposition
        weights, factors = parafac(np.nan_to_num(mapbox_tensor), rank=r, init='random', tol=10e-2, mask=~np.isnan(mapbox_tensor))
        
        # reconstruction
        fac_ls = tl.cp_to_tensor((weights, factors), mask = ~np.isnan(mapbox_tensor))
        print(fac_ls.shape)
        
        # tally computational time
        tt_ls[ii] = time.time() - tic
        print('Execution time: ' + str("{:.2f}".format(tt_ls[ii])) + ' ' + 'seconds')

        # Calculate the error of both decompositions
        # estimation error
        #est_err = tl.cp_to_tensor(fac_ls) - mapbox_tensor

        #err_ls[ii] = RMSE(np.nan_to_num(mapbox_tensor), fac_ls)
        #err_ls[ii] = tl.norm(tl.cp_to_tensor(fac_ls, mask=~np.isnan(mapbox_tensor)) - np.nan_to_num(mapbox_tensor))
        my_rmse = RMSE(np.nan_to_num(mapbox_tensor), fac_ls)
        err_ls[ii] = my_rmse
        print(f"Estimated RSME: {err_ls[ii]} == {my_rmse}")
        print("__________________________________________")    

        # frob norm of reconstruction difference (RSS)
        x_r = tl.norm(np.nan_to_num(mapbox_tensor) - np.nan_to_num(fac_ls))
        print(f"Estimated x_r: {x_r}")
        exp_var = 1 - (x_r/x_c)
        var_ls[ii] = 1 - (x_r/x_c)
        print(f"Explained variance: {var_ls[ii]} == {exp_var}")
        print("__________________________________________")

        # save weights to a file
        np.save(dir_to_save_factors + f"rank_{r}_weights", weights)

        # save factors to file 
        #for i, (w, fct) in enumerate(zip(weights, factors)):
        for i, fct in enumerate(factors):
            np.save(dir_to_save_factors + f"rank_{r}_factor_{i}", fct)


        # get rid of temp vars and collect garbage 
        del fac_ls, weights, factors
        gc.collect()
        
        print(f"weights and factors saved")
        print("===========loop is over===========")
            

    print("__________________________________________")
    print("finished calculating the CP decompositions")
    print("__________________________________________")
    print("now saving errors, variances and ranks")
    print("__________________________________________")

    
    # save errors 
    with open('/home/barguzin/gits/mapbox_decomp/src/err.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(err_ls)

    with open('/home/barguzin/gits/mapbox_decomp/src/vars.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(var_ls)

    with open('/home/barguzin/gits/mapbox_decomp/src/ranks.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ranks)

    #np.savetxt(dir_to_save_factors + "vars.csv", var_ls, delimiter=",")
    #np.savetxt(dir_to_save_factors + "errors.csv", err_ls, delimiter=",")
    #np.savetxt(dir_to_save_factors + "ranks.csv", ranks, delimiter=",")
    print("saved the errors")

    # plotting magic 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

    # plot ranks vs rmse
    ax1.plot(ranks, err_ls, 'o-');
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'));
    ax1.set_title("Reconstructive RMSE", fontsize=16);
    ax1.set_xlabel("Rank");
    ax1.set_ylabel("RMSE");
    for i in ranks:
        ax1.axvline(i, color='grey', ls='--'); 

    ax2.plot(ranks, var_ls, 'o-');

    ax2.set_title("Rank versus explained variance", fontsize=16);
    ax2.set_xlabel("Rank");
    ax2.set_ylabel("Explained Variance");
    for i in ranks:
        ax2.axvline(i, color='grey', ls='--'); 

    fig.savefig(path_to_save_figures)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_tensor = config['PATHS']['path_to_tensor']
    path_to_save_figures = config['PATHS']['path_to_save_figures']
    dir_to_save_factors = config['PATHS']['dir_to_save_factors']
    rank_params = [10,70,3]
    rank_approximator(path_to_tensor, rank_params, dir_to_save_factors, path_to_save_figures, testing=False)
