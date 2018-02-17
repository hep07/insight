from .tf_base_model import TFBaseModel

import tensorflow as tf
import numpy as np

class RNNModel(TFBaseModel):

    def __init__(self, state_size, keep_prob=1, **kwargs):
        self.state_size = state_size
        self.keep_prob = keep_prob
        super(type(self), self).__init__(**kwargs)

    def calculate_loss(self):
        self.data = tf.placeholder(tf.float32, name='data')
        self.given_days = tf.placeholder(tf.int32, name='given_days')
        self.days = tf.placeholder(tf.int32, name='days')
        batch_size = tf.shape(self.data)[0]
        
        cells = []
        for i in range(len(self.state_size)):
            c = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(
                    self.state_size[i],
                ),
                output_keep_prob=self.keep_prob,
            )
            if i != 0:
                c = tf.nn.rnn_cell.ResidualWrapper(c)
            cells.append(c)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        
        # ([batch_size, state_size])
        state = cell.zero_state(tf.shape(self.data)[0], dtype=tf.float32)
        # [batch_size, 1]
        last_output = tf.zeros([tf.shape(self.data)[0], 1], dtype=tf.float32)
        
        loss = tf.constant(0, dtype=tf.float32)
        step = tf.constant(0, dtype=tf.int32)
        output_ta = tf.TensorArray(size=self.days, dtype=tf.float32)
        
        def cond(last_output, state, loss, step, output_ta):
            return step < self.days
        
        def body(last_output, state, loss, step, output_ta):
            output, state = cell(last_output, state)
            output = tf.layers.dense(
                output,
                1,
                name='dense-top'
            )
            output_ta = output_ta.write(step, tf.transpose(output))
            
            last_output = tf.cond(
                step < self.given_days,
                lambda: tf.expand_dims(self.data[:,step], 1),
                lambda: output,
            )
            last_output.set_shape([None, 1])
                       
            true = tf.expand_dims(self.data[:,step], 1)
            loss = tf.cond(
                step >= self.given_days,
                lambda: loss + tf.reduce_mean(tf.abs(true - output)),
                lambda: loss
            )
            loss.set_shape([])
            
            return (last_output, state, loss, step + 1, output_ta)
        
        _, self.final_state, loss, _, output_ta = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(last_output, state, loss, step, output_ta)
        )
        
        self.preds = tf.transpose(output_ta.concat())
        self.prediction_tensors = {
            'preds': self.preds
        }
        
        loss = loss / tf.cast(self.days - self.given_days, tf.float32)
        return loss

    def predict(self, batch_size=1000):
        preds = []
        test_generator = self.reader.gen_test(batch_size)
        for i, test_batch_df in enumerate(test_generator):
            test_feed_dict = {
                getattr(self, placeholder_name, None): data
                for placeholder_name, data in test_batch_df.items() if hasattr(self, placeholder_name)
            }

            [batch_preds] = self.session.run(
                fetches=[self.preds],
                feed_dict=test_feed_dict
            )
            
            preds.append(batch_preds)
            print('batch {} processed'.format(i))

        return np.concatenate(preds, axis=0)
