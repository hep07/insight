# Forecaster

**Deep Learning Based Time Series Forecasting**

Jun Xie's Project in the Insight Artificial Intelligence Program

## What is it?

## Example
The objective of this project is to predict multiple sequences of future values, given the sequences of historical time series data.

The data I explored with are web traffic data of Wikipedia, which is an accessible large dataset. (https://www.kaggle.com/c/web-traffic-time-series-forecasting) The training dataset consists of approximately 145k time series. Each of these time series represents a number of daily views of a different Wikipedia article, starting from July 1st, 2015 up until September 10th, 2017. The goal is to forecast the daily views between September 13th, 2017 and November 13th, 2017 for each article in the test dataset. The name of the article as well as the type of traffic (all, mobile, desktop, spider) is given for each article. The y-axis is log transformed.
<p align="center">
  <img src="figures/wiki_data1.png">
</p>

The evaluation metric is symmetric mean absolute percentage error (SMAPE).

## Underlying models
Two models are implemented. A recurrent model based on [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and a temporal convolutional model inspired by [Wavenet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/).

## Motivation
Data in the form of time-dependent sequential observations emerge in many key real-world problems, including areas such as biological data, financial markets, demand and supply forecasting, signal collected in IoT and wearable devices, to audio and video processing. The tranditional methods (such as ARIMA) that are based on statistical features (Exponential Moving Average) can do pretty well in single time series prediction, but for multiple time series task, it can not generate predictions based on related time series that have similar pattern. Future values may depends on the previous value, it may also depends on other external time series that has similarity. 

Recurrent Neural networks, and other newer architectures, such as Wavenet and AttentionNet, have showed their success in audio processing and lauguage translation. It's an intersting topic to see how those deep learning networks perform on multiple time series data, given the fact that they are able to learn the similarities across different time series.

## Requirements
11 GB GPU (recommended), Python 3

Python packages:
- numpy==1.13.3
- pandas==0.21.0
- matplotlib==2.1.0
- scikit-learn==0.19.1
- lightgbm==2.0.10
- tensorflow==1.3.0