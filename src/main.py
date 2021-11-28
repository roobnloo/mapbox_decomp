import math
import numpy as np
import dask.dataframe as dd
import numpy.random
import pandas as pd
from tensorly.decomposition import parafac
import time
import configparser


def parquet_to_tensor(parquet_path, tensor_path):
    df = init_df(parquet_path)
    scu_tens = fill_tensor(df)

    print(f"Shape of tensor is {scu_tens.shape}")
    print(f"Number filled is {np.count_nonzero(~np.isnan(scu_tens))}")
    print(f"Number of NaNs is {np.count_nonzero(np.isnan(scu_tens))}")

    # save ndarray to a binary file for easy loading
    filename = "scu_mapbox_poi_agg_tens.npy"
    np.save(tensor_path + filename, scu_tens)

    # decompose(scu_tens, tensor_path, 1)
    print(f"Saved {filename} to {tensor_path}")


def decompose(scu_tens, tensor_path, r):
    """Runs CP with rank r on scu_tens. Saves factor mxs to tensor_path"""
    numpy.random.seed(594)
    weights, factors = parafac(np.nan_to_num(scu_tens), rank=r, init='random', tol=10e-6, mask=~np.isnan(scu_tens))
    for i, fct in enumerate(factors):
        np.save(tensor_path + f"factor_{i}", fct)


def init_df(path, part=None, keep_extra_columns=False):
    """Initializes a Dask df from parquet path"""
    df = dd.read_parquet(path)
    df = df.astype({'geography': 'int64', 'cnt': 'int32'})
    df.agg_day_period = dd.to_datetime(df.agg_day_period)
    return df


def fill_tensor(df: dd.DataFrame, remove_duplicates=True) -> np.ndarray:
    pd_df = df.compute()
    pd_df = pd_df.set_index(['geography', 'agg_day_period', 'cnt'])
    dupe_index = pd_df.index.duplicated(keep=not remove_duplicates)
    pd_df = pd_df[~dupe_index]
    pd_df = pd_df.sort_index()

    # print(f"Total {len(dupe)} indices with duplicate entries.")
    # print("==== The following spatio-temporal indices contain duplicate entries ====")
    # print(pd_df[dupe])
    # print("==== End list of duplicates ====")
    np_df = pd_df.to_xarray().to_array().to_numpy()
    return np.squeeze(np_df)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_poi_agg_parq = config['PATHS']['path_to_poi_agg_parquet']
    tensor_out_path = config['PATHS']['tensor_out_path']
    parquet_to_tensor(path_to_poi_agg_parq, tensor_out_path)
