import dask.dataframe as dd
from tensorly.decomposition import parafac

path_to_parquet = "../resources/scu_mapbox.parquet"


def run_me():
    scu_tens = dd.read_parquet(path_to_parquet)
    scu_tens = scu_tens.to_dask_array()
    scu_tens.compute_chunk_sizes()
    print(f"Shape of tensor is {scu_tens.shape}")
    # parafac(scu_tens, rank=3, init='random', tol=10e-6)
    print("Completed")


if __name__ == '__main__':
    run_me()
