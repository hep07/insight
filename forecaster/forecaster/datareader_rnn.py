import os
import numpy as np
import pandas as pd
from dataframe import DataFrame

class DataReader():
    def __init__(
        self, data_dir, 
        min_train_days, max_train_days, train_predict_days, train_loss_days, 
        val_days, predict_days, predict_warmup_days,
        seed=923
    ):
        self.data = np.load(os.path.join(data_dir, 'processed/data.npy'))
        self.fpage = pd.read_pickle(os.path.join(data_dir, 'processed/fpage.pkl'))
        self.fdate = pd.read_pickle(os.path.join(data_dir, 'processed/fdate.pkl'))
        self.seed = seed
        self.days = self.data.shape[1]
        
        self.max_train_days = max_train_days
        self.min_train_days = min_train_days
        self.train_predict_days = train_predict_days
        self.train_loss_days = train_loss_days
        self.val_days = val_days 
        self.predict_days = predict_days
        self.predict_warmup_days = predict_warmup_days
        
        self.domain = self.fpage.domain.values
        self.access = self.fpage.access.values
        self.agent = self.fpage.agent.values
        
        self.dayofweek = self.fdate.dayofweek.values
        self.isweekday = self.fdate.isweekday.values
        self.month = self.fdate.month.values
        
        
    def describe(self, logger):
        logger.info('')
        logger.info('Data dimensions:')
        logger.info('    [[data]] {}'.format(self.data.shape))
        logger.info('Split seed = {}'.format(self.seed))
        logger.info('')

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            self.data,
            batch_size=batch_size,
            start_day = 0,
            total_days = self.days - self.val_days,
            min_days = self.min_train_days,
            max_days = self.max_train_days,
            predict_days = self.train_predict_days,
            loss_days = self.train_loss_days,
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            self.data,
            batch_size=batch_size,
            start_day = 0,
            total_days = self.days,
            min_days = self.days,
            max_days = self.days,
            predict_days = self.val_days,
            loss_days = self.val_days,
        )
    
    def fit_transform(self, data, given_days=None):
        if given_days is None:
            given_days = data.shape[1]
        sc_std = StandardScaler(with_mean = False)
        sc_mean = StandardScaler(with_std = False)
        
        ref = data[:,:given_days].T
        sc_std.fit(ref)
        d = sc_std.transform(data.T)
        
        d = np.log1p(d)
        
        ref = d[:given_days, :]
        sc_mean.fit(ref)
        d = sc_mean.transform(d)
        
        return (d.T, sc_std, sc_mean)
    
    def transform(self, data, sc_std, sc_mean):
        d = sc_std.transform(data.T)
        d = np.log1p(d)
        d = sc_mean.transform(d)
        return d.T
    
    def inverse_transform(self, data, sc_std, sc_mean):
        d = sc_mean.inverse_transform(data.T)
        d = np.expm1(d)
        d = sc_std.inverse_transform(d)
        return d.T

    def test_batch_generator(self, batch_size):
        start = 0
        n = self.data.shape[0]
        while start < n:
            batch = {}
            idx = [i for i in range(start, min(start + batch_size, n))]
            batch['data'], batch['sc_std'], batch['sc_mean'] = self.fit_transform(self.data[idx, :self.predict_warmup_days])
            batch['given_days'] = self.predict_warmup_days
            batch['no_loss_days'] = self.predict_warmup_days
            batch['days'] = self.predict_warmup_days + self.predict_days
            
            batch['dayofweek'] = self.dayofweek[:batch['days']]
            batch['isweekday'] = self.isweekday[:batch['days']]
            batch['month'] = self.month[:batch['days']]
            
            batch['domain'] = self.domain[idx]
            batch['agent'] = self.agent[idx]
            batch['access'] = self.access[idx]
            
            yield batch
            
            start += batch_size

    def batch_generator(self, data, batch_size, start_day, total_days, max_days, min_days, predict_days, loss_days):
        while True:
            idx = np.random.randint(0, data.shape[0], [batch_size])
            start = np.random.randint(start_day, start_day + total_days - min_days + 1)
            days = np.random.randint(min_days, min(max_days, start_day + total_days - start) + 1)
            days_idx = [i for i in range(start, start + days)]
            given_days = days - predict_days
            no_loss_days = days - loss_days
            
            batch = {}
            batch['data'], batch['sc_std'], batch['sc_mean'] = self.fit_transform(data[idx, :][:, days_idx], no_loss_days)
            
            batch['given_days'] = given_days
            batch['no_loss_days'] = no_loss_days
            batch['days'] = days
            
            batch['dayofweek'] = self.dayofweek[days_idx]
            batch['isweekday'] = self.isweekday[days_idx]
            batch['month'] = self.month[days_idx]
            
            batch['domain'] = self.domain[idx]
            batch['agent'] = self.agent[idx]
            batch['access'] = self.access[idx]
            
            yield batch