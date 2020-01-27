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

# Welcome to Time Series! #

In this micro-course you'll learn an invaluble skill: how to predict the future!

After completing this micro-course, you'll be able to:
- Forecast the trend of a search term with linear regression.
- Predict the daily page-views of a website with [Prophet](https://facebook.github.io/prophet/).
- Estimate market demand for a ride-sharing company with XGBoost.
- Find highly-profitable customers with a Markov model.
- Build deep learning models to handle even the most complex data sets.



You'll be prepared for this micro-course if you know how to [construct a machine learning model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model), [manipulate dataframes with Pandas](https://www.kaggle.com/residentmario/indexing-selecting-assigning), and [use seaborn to explore your data](https://www.kaggle.com/alexisbcook/hello-seaborn). You'll have a leg up if you've done some work on the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition, but we'll review what we need as we go.

![Image](/images/prophet.png)

# What is a Time Series? #

A **time series** is simply a sequence of observations together with the times those observations occured. The times provide an **index** to the observations. Usually, the observations will have been made over some fixed time interval, like every hour or every month.

[Data](https://trends.google.com/trends/explore?date=2015-01-25%202020-01-25&geo=US&q=data%20science) from Google Trends. This dataset shows the popularity of the search term `data science` from January 25, 2015 to January 25, 2020, with observations taken weekly.

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()

# Read the file into a variable trends
# Parse the "Week" column as a date and set it as the index
trends = pd.read_csv("data/datascience.csv", parse_dates=["Week"], index_col="Week")

# View the first five weeks of the trends dataset
trends.head()
```

The numbers in the `Interest` column represent the popularity for that week relative to when the term was most popular over the time observed. Google says: "A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular."


# Plotting Time Series  #

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

Time series are commonly represented as [line charts](https://www.kaggle.com/alexisbcook/line-charts), with the index along the x-axis. A line chart emphasizes the ordered nature of a time series.

You can quickly plot a data frame with the `DataFrame.plot` method. For time series, `pandas` will create a line chart by default.

```python
trends.plot();
```

You learned about [seaborn](https://seaborn.pydata.org/index.html) in a previous micro-course. `seaborn` is a powerful and flexible supplement to `matplotlib` for statistical visualization. We'll use it for most of our plots.

```python
import seaborn as sns
# Change to seaborn style and set some nice defaults
sns.set()

plt.figure(figsize=(16,6))
plt.title("Popularity of 'data science' as a search term", fontweight="bold")
sns.lineplot(data=trends);
```


# Fitting a Trend-Line #

What makes a time series unique is that consecutive observations in the series will usually be *dependent*. Today tells us about tomorrow.

So, since the time index informs us about the observations, we could treat a forecasting problem as a regression problem. We could treat the series of observations as the target and the index as a feature.

To make a forecast on our series, we will fit a *linear trend-line* using [simple linear regression ](https://en.wikipedia.org/wiki/Simple_linear_regression). Our trend-line model will fit a least-squares line with `Interest` as the target and `Week` as the feature.


# Prepare Data #

There are two issues we must address when preparing our data for training. 

First, regression models require their features to be numeric. We can construct a "time dummy" for the "Weeks" column. The time dummy is just an enumeration of the observations in the time series, beginning at 1.

Second, we need to be careful when we split our data. Ordinarily, it is good practice to suffle your data set before splitting to ensure the splits are independent. But independent is exactly what the observations in a time series are not. By shuffling, we would destroy that.

In forecasting, we want to use information about the past to predict the future. We want to make sure, therefore, that all of the validation data occurs *after* the training data.

The validation set should always be later in time than the training set.

```python
from sklearn.model_selection import train_test_split

data = trends.copy()
# Construct the "time dummy": 1, 2, 3, ...
data['Week'] = range(len(data.Interest))

# Split the data into a training set and a validation set
# The order of the observations is important, so don't shuffle
train_data, val_data = train_test_split(data, test_size = 0.2, shuffle = False)
```

# Define and Fit the Model #

The `statsmodels` library is like the `sklearn` of time series. 

The easiest way to get started with `statsmodels` is through its [formula interface](https://www.statsmodels.org/stable/example_formulas.html). Formulas in `statsmodels` work the same way as formulas in R. Instead of passing in our variables as arrays (like in `sklearn`), in `statsmodels` we can specify the regression relationship with a special kind of string and let `statsmodels` create the arrays for us.

For simple linear regression, we write the formula as: `"target ~ feature"`.

```python
import statsmodels.formula.api as smf

# Fit an ordinary least-squares model using the formula interface
trends_model = smf.ols("Interest ~ Week", train_data).fit()
# Look at the fitted coefficients of the least-squares line
trends_model.params
```

The coefficients say: "Predict about 27 points of Interest for the first week, and about 0.24 more points for every week that goes by."


# Evaluate the Model #

```python
from sklearn.metrics import mean_squared_error

# In-sample predictions and RMSE
train_predictions = trends_model.predict(train_data)
rmse_train = mean_squared_error(train_data.Interest, train_predictions, squared=False)
# Out-of-sample predictions (the forecast) and RMSE
val_predictions = trends_model.predict(val_data)
rmse_val = mean_squared_error(val_data.Interest, val_predictions, squared=False)

print("RMSE of fitted predictions:")
print(rmse_train)
print()
print("RMSE of forecast predictions:")
print(rmse_val)
```

Our error increased by about 38% from the training set to the validation set. This indicates that our model may be having some trouble generalizing.

# Interpret Predictions #

Let's make a plot of our predictions to get a better idea of what's going on.

```python
plt.figure(figsize=(16,6))
sns.lineplot(data=data.Interest, alpha=0.5)
sns.lineplot(data=train_predictions, label="Fitted", color="b")
sns.lineplot(data=val_predictions, label="Forecast", color="r");
plt.title("Fitted and forecast predictions from the regression model", fontweight="bold")
plt.xlabel("Date", fontweight="bold")
plt.ylabel("Interest", fontweight="bold")
plt.legend(title="Predictions", loc="upper left")
plt.show()
```

There is an obvious seasonal ##cyclic?## component to our data. You can see that the popularity of "data science" tends to fall in the summer and winter and rise Spring and Fall. (Students on break from school?) There's information we aren't using that could help us make our predictions better. In future lessons, we'll see models that can make use of this kind of information.


# Conclusion #

The defining feature of time series is their dependence on a temporal order. This temporal dependence is both a useful source of information, but also a strong constraint. If you haven't worked with time series before, it's likely that . The models you've probably worked with before function best, work best when your applied to data that are iid.


# Your Turn #

Now you know how to make forecasts using a linear trend-line. When you're ready, move on to the first exercise!
