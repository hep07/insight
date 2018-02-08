import numpy as np
import pandas as pd
import sys, os, gc, types
from sklearn.preprocessing import LabelEncoder

root_paths = [
    "/Users/jiayou/Dropbox/JuanCode/Kaggle/Wikipedia/data2/", # Mac
    "/Users/jiayou/Dropbox/Documents/JuanCode/Kaggle/Wikipedia/data2/", # 1080
    '/Users/junxie/Dropbox/JuanCode/Insight/project/data_mini/', # pro
    '/mnt/WD Black/Dropbox/JuanCode/Insight/Project/data_mini/', # paperspace
]
root = None
for p in root_paths:
    if os.path.exists(p):
        root = p
        break
print('current working directory', root)

def parse_page(x):
    x = x.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]

df = pd.read_pickle(root + 'train_mini.pkl')
df.fillna(0, inplace = True)

# extract date features
date_cols = [i for i in df.columns if i != 'Page']

date = pd.to_datetime(date_cols)

fdate = pd.DataFrame(date_cols, columns = ['date'])

fdate['dayofweek'] = date.dayofweek
fdate['dayofmonth'] = date.day
fdate['dayofyear'] = date.dayofyear
fdate['month'] = date.month
fdate['year'] = date.year
fdate['isweekday'] = (fdate.dayofweek < 5).astype(np.int32)

fdate.drop(['date'], axis=1, inplace=True)

fdate.to_pickle(root+'processed/fdate.pkl')

# extract data
data = df[date_cols].values
np.save(root+'processed/data.npy', data)

# extract page features
df['name'], df['domain'], df['access'], df['agent'] = zip(*df['Page'].apply(parse_page))
le = LabelEncoder()
df['domain'] = le.fit_transform(df['domain'])
df['access'] = le.fit_transform(df['access'])
df['agent'] = le.fit_transform(df['agent'])
fpage = df[['domain', 'access', 'agent']]
fpage.to_pickle(root+'processed/fpage.pkl')