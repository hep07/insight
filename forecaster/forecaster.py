from .rnn_data_reader import RNNDataReader
from .rnn_model import RNNModel

class Forecaster():
	def __init__(self, data, forecast_length):
		self.data = data
		self.forecast_length = forecast_length

	def fit(self):
		self.model = self.get_model()
		self.model.fit()

	def predict(self):
		return self.model.predict()[:, -self.forecast_length:]

	def get_model(self):
		n = len(self.data)
		batch_size = 128
		step_per_epoch = int(n / batch_size)

		return RNNModel(
			state_size=[300],
	        reader=self.get_data_reader(),
	        batch_size=batch_size,
	        num_training_steps=100 * step_per_epoch,
	        learning_rate=.01,
	        lr_schedule=None,
	        optimizer='adam',
	        grad_clip=5,
	        regularization_constant=0.0,
	        early_stopping_steps=5 * step_per_epoch,
	        warm_start_init_step=0,
	        num_restarts=3,
	        enable_parameter_averaging=False,
	        min_steps_to_checkpoint=step_per_epoch,
	        log_interval=step_per_epoch,
	        loss_averaging_window=step_per_epoch,
	        work_dir='tf-data',
	        name='rnn',
		)

	def get_data_reader(self):
		return RNNDataReader(
		    data=self.data, 
		    forecast_length=self.forecast_length,
	        seed=923,
		)

