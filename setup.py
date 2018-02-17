from setuptools import setup

setup(name='forecaster',
      version='0.2',
      description='time series forecasting',
      url='https://github.com/kylinorange/insight/',
      author='Jun Xie',
      author_email='kylinorange@gmail.com',
      license='BSD',
      packages=['forecaster'],
      install_requires=[
          'numpy',
          'pandas',
          'tensorflow',
          'sklearn',
      ],
      zip_safe=False)