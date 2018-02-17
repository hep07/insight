import numpy as np

class RNNDataReader():
    def __init__(self, data, forecast_length, seed=923):
        self.data = data
        self.seed = seed
        self.days = self.data.shape[1]
        self.forecast_length = forecast_length
        
    def describe(self, logger):
        logger.info('')
        logger.info('Data dimensions:')
        logger.info('    [[data]] {}'.format(self.data.shape))
        logger.info('Split seed = {}'.format(self.seed))
        logger.info('')

    def gen_train(self, batch_size):
        return self.batch_generator(
            self.data,
            batch_size=batch_size,
            total_days = self.days - self.forecast_length,
            min_days = self.forecast_length * 2,
            predict_days = self.forecast_length,
        )

    def gen_val(self, batch_size):
        return self.batch_generator(
            self.data,
            batch_size=batch_size,
            total_days = self.days,
            min_days = self.days,
            predict_days = self.forecast_length,
        )
    
    def gen_test(self, batch_size):
        start = 0
        n = self.data.shape[0]
        while start < n:
            batch = {}
            idx = [i for i in range(start, min(start + batch_size, n))]
            batch['data'] = self.data[idx, :self.days]
            batch['given_days'] = self.days
            batch['days'] = self.days + self.forecast_length
            
            yield batch
            
            start += batch_size

    def batch_generator(self, data, batch_size, total_days, min_days, predict_days):
        while True:
            idx = np.random.randint(0, data.shape[0], [batch_size])
            start = np.random.randint(0, total_days - min_days + 1)
            days = np.random.randint(min_days, total_days - start + 1)
            days_idx = [i for i in range(start, start + days)]
            given_days = days - predict_days
            
            batch = {}
            batch['data'] = data[idx, :][:, days_idx]            
            batch['given_days'] = given_days
            batch['days'] = days
            
            yield batch