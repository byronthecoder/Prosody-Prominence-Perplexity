import os
import glob
import pandas as pd
import numpy as np


pf_path = sorted(glob.glob('/Users/the-imitation-gamer/Documents/SLP/Msc_Dissertation/data/pf_means/*/*.means'))
pfs_df = []
for p in pf_path:
    pf_df = pd.read_csv(p, sep="\s+", index_col=0, engine='python')

    special_characters = pf_df.index.str.extractall(r'(?P<square>.*\[.*)')

    pf_df.drop(special_characters.square, inplace=True)

    # pf_df.replace('--undefined--', np.nan, inplace=True)

    # pf_df = pf_df.astype("float64")

    pfs_df.append(pf_df.index)
p_features = pd.concat(pfs_df)

p_features.to_csv('/Users/the-imitation-gamer/Documents/SLP/Msc_Dissertation/data/csv/swb_no_special_char.csv')