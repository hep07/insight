import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import logging
import os, imp, sys
import pprint as pp

import numpy as np
import tensorflow as tf

def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]

class TFBaseModel(object):

    """Interface containing some boilerplate code for training tensorflow models.
    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.
    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
    """

    def __init__(
        self,
        reader,
        batch_size=128,
        num_training_steps=20000,
        learning_rate=.01,
        lr_schedule=None,
        optimizer='adam',
        grad_clip=5,
        regularization_constant=0.0,
        early_stopping_steps=3000,
        warm_start_init_step=0,
        num_restarts=None,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=100,
        log_interval=20,
        loss_averaging_window=100,
        work_dir='tf-data',
        name='nn'
    ):

        self.reader = reader
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.early_stopping_steps = early_stopping_steps if early_stopping_steps is not None else np.inf
        self.enable_parameter_averaging = enable_parameter_averaging
        self.num_restarts = num_restarts
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.loss_averaging_window = loss_averaging_window
        self.name = name

        self.log_dir = os.path.join(work_dir, 'logs')
        self.prediction_dir = os.path.join(work_dir, 'predictions')
        self.checkpoint_dir = os.path.join(work_dir, 'checkpoints')
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = os.path.join(work_dir, 'checkpoints-avg')

        self.init_logging(self.log_dir)
        self.logger.info('\nNetwork hyper-parameters:\n{}'.format(pp.pformat(self.__dict__)))
        # self.logger.info('\nData reader parameters:\n{}'.format(pp.pformat(self.reader.__dict__)))
        self.reader.describe(self.logger)

        self.graph = self.build_graph()
        self.session = tf.Session(graph=self.graph)
        print('Built graph')

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    def before_step(self, step):
        pass

    def after_step(self, step):
        pass

    def set_learning_rate(self, step):
        if self.lr_schedule is not None:
            for s, lr in self.lr_schedule:
                if s == step:
                    self.learning_rate = lr
                    break

    def fit(self):
        with self.session.as_default():

            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                self.session.run(self.init)
                step = 0

            train_generator = self.reader.gen_train(self.batch_size)
            val_generator = self.reader.gen_val(self.batch_size)

            train_loss_history = deque(maxlen=self.loss_averaging_window)
            val_loss_history = deque(maxlen=self.loss_averaging_window)

            smoothed_tlh = []
            smoothed_vlh = []

            best_validation_loss, best_validation_tstep = float('inf'), None
            restarts = 0

            while step < self.num_training_steps:
                self.before_step(step)
                self.set_learning_rate(step)

                if step > 0 and step % self.loss_averaging_window == 0:
                    plt.figure(figsize=(20, 5))
                    plt.plot(smoothed_tlh, 'k:', label='train loss')
                    plt.plot(smoothed_vlh, 'g', label='val loss')
                    plt.legend()
                    plt.show()

                # validation evaluation
                val_batch_df = next(val_generator)
                val_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in val_batch_df.items() if hasattr(self, placeholder_name)
                }

                val_feed_dict.update({self.learning_rate_var: self.learning_rate})
                [val_loss] = self.session.run(
                    fetches=[self.loss],
                    feed_dict=val_feed_dict
                )
                val_loss_history.append(val_loss)

                # train step
                train_batch_df = next(train_generator)
                train_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in train_batch_df.items() if hasattr(self, placeholder_name)
                }

                train_feed_dict.update({self.learning_rate_var: self.learning_rate})
                train_loss, _ = self.session.run(
                    fetches=[self.loss, self.step],
                    feed_dict=train_feed_dict
                )
                train_loss_history.append(train_loss)

                if step > 0 and (step % self.log_interval == 0 or step + 1 == self.num_training_steps):
                    # print loss
                    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    smoothed_tlh.append(avg_train_loss)
                    smoothed_vlh.append(avg_val_loss)
                    metric_log = (
                        "[[step {:>8}]]     "
                        "[[train]]     loss: {:<12}     "
                        "[[val]]     loss: {:<12}"
                    ).format(step, round(avg_train_loss, 8), round(avg_val_loss, 8))
                    self.logger.info(metric_log)

                    if len(val_loss_history) == self.loss_averaging_window:
                        # save checkpoint
                        if step >= self.min_steps_to_checkpoint and \
                            avg_val_loss < best_validation_loss:

                            best_validation_loss = avg_val_loss
                            best_validation_tstep = step
                            self.save(step)
                            if self.enable_parameter_averaging:
                                self.save(step, averaged=True)

                        # early stopping
                        if best_validation_tstep is not None and \
                            step - best_validation_tstep >= self.early_stopping_steps:

                            if self.num_restarts is None or restarts >= self.num_restarts:
                                self.logger.info('Early stopping')
                                break

                            if restarts < self.num_restarts:
                                self.logger.info('')
                                self.restore(best_validation_tstep)
                                self.learning_rate /= 5.0
    #                             self.early_stopping_steps /= 2
                                step = best_validation_tstep
                                train_loss_history = deque(maxlen=self.loss_averaging_window)
                                val_loss_history = deque(maxlen=self.loss_averaging_window)
                                restarts += 1
                                self.logger.info(
                                    'Reduce learning rate to {} and restore step {}'.format(self.learning_rate, step)
                                )

                self.after_step(step)
                step += 1

            if step <= self.min_steps_to_checkpoint:
                best_validation_tstep = step
                self.save(step)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

            self.logger.info('Training ended')
            self.logger.info(
                'Best validation loss of {} at training step {}'.format(
                    round(best_validation_loss, 8),
                    best_validation_tstep
                )
            )

    def predict(self, batch_size=128, num_batches=None, data_gen=None):
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)

        preds = {}
        if hasattr(self, 'prediction_tensors'):
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}

            if data_gen is None:
                data_gen = self.reader.gen_test(batch_size)

            for i, test_batch_df in enumerate(data_gen):
                test_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in test_batch_df.items() if hasattr(self, placeholder_name)
                }

                tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                np_tensors = self.session.run(
                    fetches=tf_tensors,
                    feed_dict=test_feed_dict
                )
                for tensor_name, tensor in zip(tensor_names, np_tensors):
                    prediction_dict[tensor_name].append(tensor)

                if num_batches is not None and i + 1 == num_batches:
                    break

            for tensor_name, tensor in prediction_dict.items():
                np_tensor = np.concatenate(tensor, 0)
                preds[tensor_name] = np_tensor
                # save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                # self.logger.info('Saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                # np.save(save_file, np_tensor)

        if hasattr(self, 'parameter_tensors'):
            for tensor_name, tensor in self.parameter_tensors.items():
                np_tensor = tensor.eval(self.session)
                preds[tensor_name] = np_tensor
                # save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                # self.logger.info('Saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                # np.save(save_file, np_tensor)

        return preds

    def save(self, step, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            self.logger.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        self.logger.info('saving model to {}'.format(model_path))
        saver.save(self.session, model_path, global_step=step)

    def restore(self, step=None, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not step:
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            self.logger.info('Restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                checkpoint_dir, 'model{}-{}'.format('_avg' if averaged else '', step)
            )
            self.logger.info('Restoring model from {}'.format(model_path))
            saver.restore(self.session, model_path)

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        logger = logging.getLogger(
            '{}.{}'.format(
                type(self).__name__,
                datetime.now().strftime('%Y-%m-%d.%H-%M-%S.%f')
            )
        )
        logger.setLevel(logging.INFO)
        fmtr = logging.Formatter(
            fmt='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        h = logging.StreamHandler(stream=sys.stdout)
        h.setFormatter(fmtr)
        logger.addHandler(h)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log.{}.{}.{}.txt'.format(type(self).__name__, self.name, date_str)
        h = logging.FileHandler(filename=os.path.join(log_dir, log_file))
        h.setFormatter(fmtr)
        logger.addHandler(h)

        self.logger = logger

    def update_parameters(self, loss):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_var = tf.Variable(0.0, trainable=False)

        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
            loss = loss + self.regularization_constant*l2_norm

        optimizer = self.get_optimizer(self.learning_rate_var)
        grads = optimizer.compute_gradients(loss)
        clipped = [
            (g if g is None else tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in grads
        ]

        step = optimizer.apply_gradients(clipped, global_step=self.global_step)

        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

    def log_parameters(self):
        self.logger.info(
            (
                '\n\n'
                'All parameters:\n{}\n'
                'Trainable parameters:\n{}\n'
                'Trainable parameter count: {}\n'
            ).format(
                pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]),
                pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]),
                str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())),
            )
        )

    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, gd, or rms'

    def build_graph(self):
        with tf.Graph().as_default() as graph:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.995)

            self.loss = self.calculate_loss()
            self.update_parameters(self.loss)
            self.log_parameters()

            self.saver = tf.train.Saver(max_to_keep=1)
            if self.enable_parameter_averaging:
                self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()

            return graph