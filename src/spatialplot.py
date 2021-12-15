import numpy as np
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import json
import mercantile


def dump_quadkey_map(parquet_path, path_to_quadkey_map):
    df = dd.read_parquet(parquet_path)
    df = df.astype({'geography': 'int64'})
    df = df.drop_duplicates(subset='geography').loc[:, ["geography", "xlat", "xlon"]].compute()
    quadkey_to_coord = pd.Series(zip(df.xlat.values, df.xlon.values), index=df.geography).to_dict()
    json.dump(quadkey_to_coord, open(f"{path_to_quadkey_map}quadkey_map.json", 'w'))
    print(f"Dumped {path_to_quadkey_map}quadkey_map.json")


def read_quadkey_map(path_to_quadkey_map):
    quadkey_map = json.load(open(f"{path_to_quadkey_map}quadkey_map.json"))
    quadkey_map = {int(k): v for k, v in quadkey_map.items()}
    return quadkey_map


def create_spatial_plot(coords, rank, fact_dir, figures_dir):
    spatial_factors = np.load(fact_dir + f"rank_{str(rank)}_factor_0.npy")
    multiidx = pd.MultiIndex.from_tuples(coords)
    dominant = get_movement_matrix(spatial_factors, 0, multiidx)
    subdominant = get_movement_matrix(spatial_factors, 1, multiidx)
    thirddominant = get_movement_matrix(spatial_factors, 2, multiidx)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle(f"Rank {rank} decomposition")
    ax1.pcolormesh(dominant)
    ax1.set_title("Factor 1")
    ax2.pcolormesh(subdominant)
    ax2.set_title("Factor 2")
    ax3.pcolormesh(thirddominant)
    ax3.set_title("Factor 3")
    plt.savefig(f"{figures_dir}spatialplot_rank{rank}")
    plt.clf()


def get_movement_matrix(spatial_factors, factor_idx, multi_idx):
    mx = spatial_factors[:, factor_idx]
    mx = pd.Series(mx, index=multi_idx).to_xarray().to_numpy()
    return mx


def get_coordinates(dir_parquet):
    df = dd.read_parquet(dir_parquet)
    sorted_qks = df.geography.unique().compute().sort_values()
    coords = [get_coord(g) for g in sorted_qks]
    return sorted_qks, coords


def get_coord(qk):
    bounds = mercantile.bounds(mercantile.quadkey_to_tile(qk))
    return (bounds.north + bounds.south)/2, (bounds.east + bounds.west)/2


def serialize_data(qks, rank, fact_dir, dir_save_figures):
    spatial_factors = np.load(fact_dir + f"rank_{str(rank)}_factor_0.npy")
    f0 = spatial_factors[:, 0]
    f1 = spatial_factors[:, 1]
    f2 = spatial_factors[:, 2]
    s0 = pd.Series(f0)
    s1 = pd.Series(f1)
    s2 = pd.Series(f2)
    geography = pd.Series(qks)
    df = pd.concat([geography, s0, s1, s2], axis=1)
    df.to_csv(f"{dir_save_figures}spatial_factors_rank_{rank}.csv")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    dir_to_save_factors = config['PATHS']['dir_to_save_factors']
    dir_save_figures = config['PATHS']['path_to_save_figures']
    dir_parquet = config['PATHS']['path_to_poi_agg_parquet']
    qks, coordinates = get_coordinates(dir_parquet)
    for rank in range(22, 71, 3):
        # create_spatial_plot(coordinates, rank, dir_to_save_factors, dir_save_figures)
        serialize_data(qks, rank, dir_to_save_factors, dir_save_figures)
