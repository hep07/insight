from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_page(x):
    x = x.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]


def nan_fill_forward(x):
    for i in range(x.shape[0]):
        fill_val = None
        for j in range(x.shape[1] - 3, x.shape[1]):
            if np.isnan(x[i, j]) and fill_val is not None:
                x[i, j] = fill_val
            else:
                fill_val = x[i, j]
    return x

root = 'tf-data/'

df = pd.read_csv('data_mini/train_mini.csv', encoding='utf-8')
date_cols = [i for i in df.columns if i != 'Page']

df['name'], df['project'], df['access'], df['agent'] = zip(*df['Page'].apply(parse_page))

le = LabelEncoder()
df['project'] = le.fit_transform(df['project'])
df['access'] = le.fit_transform(df['access'])
df['agent'] = le.fit_transform(df['agent'])
df['page_id'] = le.fit_transform(df['Page'])

if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')

df[['page_id', 'Page']].to_csv(root+'processed/page_ids.csv', encoding='utf-8', index=False)

data = df[date_cols].values
np.save(root+'processed/data.npy', np.nan_to_num(data))
np.save(root+'processed/is_nan.npy', np.isnan(data).astype(int))
np.save(root+'processed/project.npy', df['project'].values)
np.save(root+'processed/access.npy', df['access'].values)
np.save(root+'processed/agent.npy', df['agent'].values)
np.save(root+'processed/page_id.npy', df['page_id'].values)

test_data = nan_fill_forward(df[date_cols].values)
np.save(root+'processed/test_data.npy', np.nan_to_num(test_data))
np.save(root+'processed/test_is_nan.npy', np.isnan(test_data).astype(int))