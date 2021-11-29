import numpy as np
import dask.dataframe as dd
import pandas as pd
import tensorly as tl
import matplotlib.pyplot as plt
import configparser
import json


def dump_quadkey_map(parquet_path, path_to_quadkey_map):
    df = dd.read_parquet(parquet_path)
    df = df.astype({'geography': 'int64'})
    df = df.drop_duplicates(subset='geography').loc[:, ["geography", "xlat", "xlon"]].compute()
    quadkey_to_coord = pd.Series(zip(df.xlat.values, df.xlon.values), index=df.geography).to_dict()
    json.dump(quadkey_to_coord, open(f"{path_to_quadkey_map}quadkey_map.json", 'w'))
    print(f"Dumped {path_to_quadkey_map}quadkey_map.json")


def read_quadkey_map(path_to_quadkey_map):
    quadkey_map = json.load(open(f"{path_to_quadkey_map}quadkey_map.json"))
    return quadkey_map


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    dir_to_save_factors = config['PATHS']['dir_to_save_factors']
    dir_save_figures = config['PATHS']['path_to_save_figures']
    parquet_dir = config['PATHS']['path_to_scu_parquet']
    dump_quadkey_map(parquet_dir, dir_save_figures)
