---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


# Time Series in Python #

Learn about time series models, how to make forecasts, and effective ways of dealing with time dependent data.

## Outline ##

- Your First Model
- Tricks for Time Series in Pandas
- Exploring Time Series
- Regression Models
- Exponential Smoothing Models
- Evaluating Models
- Resampling and Cross-Validation
- Additive Models and Prophet
- ARIMA Models
- State Space Models
- Lag-Embedding Models
- Deep Learning Models with Keras
- Sequential Learning, Clustering, and Classification

# Notes

There are several kinds of problems depending on what data is sequence dependent: features, response, or both.

## Forecasting ##

In this problem, all time-dependent data is unknown past a certain date. We are trying to predict some time-dependent variable of interest. Example: forecasting stock prices.

## Sequential Supervised Learning ##

In this problem, we are trying to predict the remainder of a partial sequence given a sequence that is complete.

Sometimes, a forecasting model will be trained on a complete predictor to evaluate the quality of the model's predictions. In this case, it is known as *ex-post* forecasting, and forecasting on the partial predictor is known as *ex-ante* forecasting.

## Predicting with Dependent Features ##


## Kaggle Sources ##

### Notebooks ###

https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series
https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru
