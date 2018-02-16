import os
import numpy as np
import pandas as pd
from dataframe import DataFrame

class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'data',
            'is_nan',
            'page_id',
            'project',
            'access',
            'agent',
            'test_data',
            'test_is_nan'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

    
    def describe(self, logger):
        logger.info('')
        logger.info('Data dimensions:')
        logger.info('train size', len(self.train_df))
        logger.info('val size', len(self.val_df))
        logger.info('test size', len(self.test_df))
        # logger.info('    [[data]] {}'.format(self.data.shape))
        #logger.info('Split seed = {}'.format(self.seed))
        logger.info('')

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )
        data_col = 'test_data' if is_test else 'data'
        is_nan_col = 'test_is_nan' if is_test else 'is_nan'
        for batch in batch_gen:
            num_decode_steps = 64
            full_seq_len = batch[data_col].shape[1]
            max_encode_length = full_seq_len - num_decode_steps if not is_test else full_seq_len

            x_encode = np.zeros([len(batch), max_encode_length])
            y_decode = np.zeros([len(batch), num_decode_steps])
            is_nan_encode = np.zeros([len(batch), max_encode_length])
            is_nan_decode = np.zeros([len(batch), num_decode_steps])
            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])

            for i, (seq, nan_seq) in enumerate(zip(batch[data_col], batch[is_nan_col])):
                rand_len = np.random.randint(max_encode_length - 365 + 1, max_encode_length + 1)
                x_encode_len = max_encode_length if is_test else rand_len
                x_encode[i, :x_encode_len] = seq[:x_encode_len]
                is_nan_encode[i, :x_encode_len] = nan_seq[:x_encode_len]
                encode_len[i] = x_encode_len
                decode_len[i] = num_decode_steps
                if not is_test:
                    y_decode[i, :] = seq[x_encode_len: x_encode_len + num_decode_steps]
                    is_nan_decode[i, :] = nan_seq[x_encode_len: x_encode_len + num_decode_steps]

            batch['x_encode'] = x_encode
            batch['encode_len'] = encode_len
            batch['y_decode'] = y_decode
            batch['decode_len'] = decode_len
            batch['is_nan_encode'] = is_nan_encode
            batch['is_nan_decode'] = is_nan_decode

            yield batch