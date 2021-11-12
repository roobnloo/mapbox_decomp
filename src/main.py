import math
import numpy as np
import dask.dataframe as dd
from tensorly.decomposition import parafac

path_to_parquet = "../resources/scu_mapbox.parquet"


def main():
    df = init_df(path_to_parquet)
    # print(df.head())

    lats = df.sort_values("xlat").xlat.unique().compute().tolist()
    lons = df.sort_values("xlon").xlon.unique().compute().tolist()
    days = df.sort_values('agg_day_period').agg_day_period.unique().compute().tolist()

    # Initialize tensor to be all missing values (nan)
    scu_tens = np.empty((len(lats), len(lons), len(days)))
    scu_tens[:] = np.nan

    fill_tensor(df, lats, lons, days, scu_tens)

    print(f"Shape of tensor is {scu_tens.shape}")
    print(f"Number filled is {np.count_nonzero(~np.isnan(scu_tens))}")
    print(f"Number of NaNs is {np.count_nonzero(np.isnan(scu_tens))}")

    # save ndarray to a binary file for easy loading
    np.save("../resources/scu_tens.npy", scu_tens)

    # parafac(scu_tens, rank=3, init='random', tol=10e-6)
    print("Completed")


def init_df(path):
    """Initializes a Dask df from parquet path"""
    scu_tens_parq = dd.read_parquet(path)
    scu_tens_parq = scu_tens_parq[["xlat", "xlon", "agg_day_period", "activity_index_total"]]
    df = scu_tens_parq.partitions[0]
    df.agg_day_period = dd.to_datetime(df.agg_day_period)
    return df


def fill_tensor(df, lats, lons, days, scu_tens):
    """Fills in the values of scu_tens from dataframe df. This code is currently hot garbage. Very slow."""
    for i, xlat in enumerate(lats):
        print(f"iter {i}")
        print(f"Number filled: {np.count_nonzero(~np.isnan(scu_tens))}")
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
    main()
