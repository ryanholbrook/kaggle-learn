---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Lesson 1 - Your First Model ##

In this micro-course, you'll learn about modeling *time series*. Time series are common and important. Many Kaggle competitions have used time series.

![Image](https://image)

After completing this course, you'll be able to answer questions like:
- How many things will someone buy?
- How many passengers will an airline have?
- When will disasters occur?
- What is trending on Google?
- How many hits will a website have?

You'll visualize a trend with a word, model this thing with this, do some cool deep learning with Keras, and do state-of-the-art forecasting with Facebook's Prophet.

For your first lesson, you'll build a linear regression to forecast a trend. Be sure to review *these previous lessons* if you're feeling rusty.


```python
%matplotlib inline
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
```


### What is a Time Series? ###

A **time series** is a sequence of *observations* together with the *times* when those observations were taken. The sequence of times is called the *index* of the series.

Here, for instance, is a portion of the `sunspots` dataset from the `statsmodels` package.

```python jupyter={"source_hidden": true}
sunspots = sm.datasets.sunspots.load_pandas().data
sunspots.YEAR = pd.to_datetime(sunspots.YEAR, format='%Y')
sunspots.set_index('YEAR', inplace=True)
sunspots.head()
```

The `YEAR` column is the index and the `SUNACTIVITY` column is the sequence of observations for those years. A time series always has at least two dimensions.

Time series are commonly plotted as line graphs, with the index along the x-axis. This helps us to see how the observations are changing over time.

```python jupyter={"source_hidden": true}
sunspots.plot(figsize=(8,6));
```

What makes time series unique is that consecutive observations in the series will usually be *dependent*. Each observation will determine, to an extent, the observation that comes next. How the weather is today will affect how the weather is tomorrow.

This temporal dependence is both a useful source of information, but also a strong constraint. Most of the methods you've used in previous courses ...

### The Data ###

In this lesson, we will use linear regression to model a relationship persisting over time. The data set is here. Let's load it.

```python
icecream = sm.datasets.get_rdataset("Icecream", "Ecdat").data
icecream = icecream.set_index(pd.date_range(start='03/18/1951', freq='4W', periods=len(icecream)))
icecream.head()
```

We might guess that there could be a relationship between the temperature and the amount of icecream consumed.

```python
ax = icecream[['temp', 'cons']].plot(secondary_y = 'cons', figsize=(8,6))
ax.set_ylabel('Average Temperature (in Fahrenheit)')
ax.right_ax.set_ylabel('Consumption of Icecream per head (in pints)');
```

Now we'll define the prediction target and the features. 

```python
y = icecream.cons.values
x = icecream.temp.values.reshape(-1, 1)
```

Define the model.

```python
from sklearn.linear_model import LinearRegression

icecream_model = LinearRegression()
icecream_model.fit(x, y)

icecream_model.intercept_
icecream_model.coef_
```


### Training the Model ###

Use linear regression. Split the data and train.

```python

```

Make a plot

```python

```

### Forecasting ###

Now use the model to make a forecast.

```python

```

Plot the result.

```python

```

Evaluate the forecast.

```python

```


### Your Turn ###

You learned how to do forecasts with linear regression. When you're ready, move on to the first exercise!

# Draft

Ordinary methods of prediction (like linear regression or boosting) work best when the training set closely resembles the test set. This means that it is important to make sure your data is randomized before splitting it in any way, like for cross-validation. You want to shuffle a deck of cards before dealing to deal fair hands.

A time series, however, must be kept in order. The *order* of the time series is what encodes its time-dependent information. Shuffling the time series would destroy that. If you shuffle a time series of daily temperatures for a year, you no longer have seasons.

All of the practices you have learned in previous lessons -- like model validation and exploratory analysis -- are still important, but they will have to be modified to account for this time dependence. Some of the models you have learned can also be used to make predictions with time series, but we will develop several new models designed especially to make use of the time dependence.


```python
longley = sm.datasets.longley.load_pandas().data
longley.YEAR = pd.to_datetime(longley.YEAR, format='%Y')
longley.set_index('YEAR', inplace=True)
longley.head()
```

```python
orange = sm.datasets.get_rdataset("Orange", "Ecdat").data
orange.set_index(pd.date_range(start='01/1948', freq='M', periods=642))
orange.head()
```
