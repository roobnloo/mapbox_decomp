import math
import numpy as np
import dask.dataframe as dd
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
    np.save(tensor_path + "scu_tens.npy", scu_tens)

    decompose(scu_tens, tensor_path, 4)
    print("Completed")


def decompose(scu_tens, tensor_path, r):
    weights, factors = parafac(scu_tens, rank=r, init='random', tol=10e-6, mask=~np.isnan(scu_tens))
    for i, fct in enumerate(factors):
        np.save(tensor_path + f"factor_{i}", fct)


def init_df(path, part=0, keep_extra_columns=False):
    """Initializes a Dask df from parquet path"""
    scu_tens_parq = dd.read_parquet(path)
    if not keep_extra_columns:
        scu_tens_parq = scu_tens_parq[["xlat", "xlon", "agg_day_period", "activity_index_total"]]
    df = scu_tens_parq.partitions[part]
    df.agg_day_period = dd.to_datetime(df.agg_day_period)
    return df


def fill_tensor(df: dd.DataFrame, remove_duplicates=True) -> np.ndarray:
    pd_df = df.compute()
    pd_df = pd_df.set_index(['xlat', 'xlon', 'agg_day_period'])
    dupe_index = pd_df.index.duplicated(keep=not remove_duplicates)
    pd_df = pd_df[~dupe_index]
    pd_df = pd_df.sort_index()

    # print(f"Total {len(dupe)} indices with duplicate entries.")
    # print("==== The following spatio-temporal indices contain duplicate entries ====")
    # print(pd_df[dupe])
    # print("==== End list of duplicates ====")

    return pd_df.to_xarray().to_array().to_numpy()


def fill_tensor_slow(df, lats, lons, days, scu_tens):
    """Fills in the values of scu_tens from dataframe df. This code is currently hot garbage. Very slow."""
    tic = time.perf_counter()
    for i, xlat in enumerate(lats):
        toc = time.perf_counter()
        print(f"iter {i}")
        print(f"Number filled: {np.count_nonzero(~np.isnan(scu_tens))}")
        print(f"Elapsed time: {(toc - tic)/60:1.1f} minutes")
        matching_df0 = df.loc[df['xlat'] == lats[i]]
        matching_xlons = matching_df0.xlon.compute()
        for xlon in matching_xlons:
            j = lons.index(xlon)
            matching_df1 = matching_df0.loc[matching_df0['xlon'] == xlon]
            matching_dates = matching_df1.agg_day_period.compute()
            for d in matching_dates:
                k = days.index(d)
                vals = matching_df1[matching_df1['agg_day_period'] == d].activity_index_total.compute().values
                scu_tens[i, j, k] = vals[0]


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_scu_parquet = config['PATHS']['path_to_scu_parquet']
    tensor_out_path = config['PATHS']['tensor_out_path']
    parquet_to_tensor(path_to_scu_parquet, tensor_out_path)
