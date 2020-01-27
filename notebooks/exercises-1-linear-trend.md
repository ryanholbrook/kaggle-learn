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

# Setup #

In this exercise, you'll investigate the popularity trend of the search term ["machine learning"](https://trends.google.com/trends/explore?date=2015-01-25%202020-01-25&geo=US&q=machine%20learning) as given by Google Trends.

```python
# dataframes
import pandas as pd
pd.plotting.register_matplotlib_converters()
# graphics
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Read the file into a variable 'machinelearning'
# Parse the 'Week' column as a date and set it as the index
machinelearning = pd.read_csv('data/machinelearning.csv', parse_dates=['Week'], index_col='Week')

# View the first five weeks of the 'machinelearning' dataset
machinelearning.head()
```

# Plot It #

Now try plotting the time series. Try calling `dir(plt)` to see the methods available to modify the plot and look at their help file with `?plt.method`.

```python
# TODO
```

What is different about this time series from the time series for "data science"? How do you expect this will affect our predictions?

# Split the Data #

It is important that you do not shuffle time series data before splitting it. With time series, the order of our data sets must be preserved.

Split the time series into a training set and a validation set, being sure to preserve the time order.

```python
from sklearn.model_selection import train_test_split

# TODO
```

# Fit the Model #

```python
# modeling
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

# TODO
```

Give the appropriate formula for the trend-line model.

```python
# 'Interest ~ Week'
```


# Evaluate #

# Discuss #

