from tensorly.decomposition import parafac, non_negative_parafac
import numpy as np
import tensorly as tl
import configparser
from datetime import datetime

def store_factors_weights(path_to_tensor, rank_params, dir_to_save_factors, cp_type="parafac"): 
    """run parafac and save factors without any other actions"""

    mapbox_tensor = tl.tensor(np.load(path_to_tensor))
    ranks = np.arange(rank_params[0], rank_params[1], rank_params[2])

    for ii, r in enumerate(ranks):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Estimating for rank {r}: started at {now}")
        
        # decomposition
        if cp_type=='parafac': 
            print("initiating PARAFAC decomposition") 
            weights, factors = parafac(np.nan_to_num(mapbox_tensor), rank=r, init='random', tol=10e-2, mask=~np.isnan(mapbox_tensor))
            print(f"Decomposition complete. Now saving")
        
        else: 
            print("initiating Non-Negative PARAFAC decomposition") 
            weights, factors = non_negative_parafac(np.nan_to_num(mapbox_tensor), rank=r, init='random', tol=10e-2, mask=~np.isnan(mapbox_tensor))
            print(f"Decomposition complete. Now saving")

        np.save(dir_to_save_factors + f"rank_{r}_weights", weights)

        for i, fct in enumerate(factors):
            np.save(dir_to_save_factors + f"rank_{r}_factor_{i}", fct)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_tensor = config['PATHS']['path_to_tensor']
    dir_to_save_factors = config['PATHS']['dir_to_save_factors']
    rank_params = [10,71,3]
    #store_factors_weights(path_to_tensor, rank_params, dir_to_save_factors, cp_type="NNCP")
    store_factors_weights(path_to_tensor, rank_params, dir_to_save_factors)
