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

Try plotting the time series. Use the default `pandas` methods or experiment with `seaborn`. Try calling `dir(plt)` to see the methods available to modify the plot and look at help files with `?plt.method`.

```python

```

What is different about this time series from the time series for "data science"? How do you expect this will affect your predictions?

# Create a Time Dummy #

Now get your data ready for modeling. First create a time dummy from the `Week`.

```python
# Your code here
machinelearning['Week'] = ____
```

# Create and Evaluation Split #

Now Split the time series into a training set and a validation set, being sure to preserve the time order.

```python
from sklearn.model_selection import train_test_split

# Your code here
train_data, val_data = ____
```

# Specify Model #

Check your understanding of formulas. Give the appropriate formula for the trend-line model.

```python
# Your code here
formula = ____ # 'Interest ~ Week'
```

# Fit Model #

Define a linear regression model using the formula interface to `statsmodels`.

```python
import statsmodels.formula.api as smf

# Your code here
machinelearning_model = ____
```

# Evaluate #

Make predictions from the fitted model and evaluate its performance.

```python
from sklearn.metrics import mean_squared_error

# Your code here
train_predictions = ____
rmse_train = ____
val_predictions = ____
rmse_val = ____
```

# Discuss #

One thing we haven't considered in this lesson was the nature of the popularity score. For one, the score has to be within 0 and 100 if we are to preserve the original interpretation. But a linear-trend model could return scores outside of this range. What would we do with a forecast of -20? Of 175? We might also wonder whether RMSE is the right error metric to use for this kind of data.

# Keep Going! #

You're doing great! Move on to the next lesson and learn about how to break time series down into simpler parts.
