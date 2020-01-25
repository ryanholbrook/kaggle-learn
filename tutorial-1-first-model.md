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

In this micro-course, you'll learn about modeling *time series*. Time series are common and important. Many Kaggle competitions have used time series.

![Image](https://image)

After completing this course, you'll be able to answer questions like:
- How many things will someone buy?
- How many passengers will an airline have?
- When will disasters occur?
- What is trending on Google?
- How many hits will a website have?

You'll visualize a trend with a word, model this thing with this, do some cool deep learning with Keras, and do state-of-the-art forecasting with Facebook's Prophet.

For your first lesson, you'll build a linear regression to forecast a trend. 

You'll be prepared for this micro-course if you know how to [construct a machine learning model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model), [manipulate dataframes with Pandas](https://www.kaggle.com/residentmario/indexing-selecting-assigning), and [use seaborn to explore your data](https://www.kaggle.com/alexisbcook/line-charts) Be sure to review *these previous lessons* if you're feeling rusty.


# What is a Time Series? #

A **time series** is simply a sequence of observations together with the times those observations occured. The times provide an **index** to the observations. Usually, the observations will have been made over some fixed time interval, like every hour or every month.

[Data](https://trends.google.com/trends/explore?date=2015-01-25%202020-01-25&geo=US&q=data%20science) from Google Trends. This dataset shows the popularity of the search term `data science` from January 25, 2015 to January 25, 2020, with observations taken weekly.

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()

# Read the file into a variable trends
# Use parse_dates on any datetime columns
trends = pd.read_csv("data/trends.csv", parse_dates = ["Week"])

# View the first five weeks of the trends dataset
trends.head()
```

The numbers in the `Interest` column represent the popularity for that week relative to when the term was most popular in the time given. Google says: "A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular."

We can use the `set_index` method of `DataFrame` to make the `Week` column in `trends` the index column.

```python
# Replace the index of the dataframe with the "Week" column
# Use inplace to modify destructively instead of making a copy
trends.set_index('Week', inplace = True)

# View the first five weeks again. Now "Week" is the index
trends.head()
```

We could also have set the index column when we read the data by using the `index_col` argument in `read_csv`.

# Plotting Time Series  #

```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

Time series are commonly plotted as line charts, with the index along the x-axis. A line charts makes the ordered nature of a time series more apparent. 

`pandas` offers a quick way to plot timeseries with the `plot` method. It will create a [line chart](https://www.kaggle.com/alexisbcook/line-charts) by default, though that can be changed with the `kind` parameter.

```python
trends.plot(figsize=(16,6));
```

You learned about the [seaborn](https://seaborn.pydata.org/index.html) library in a previous micro-course. It is powerful and flexible.

```python
import statsmodels.api as sm
# Change to seaborn plot style
sns.set()

plt.figure(figsize=(16,6))
plt.title("Popularity of 'data science' as a search term", fontweight="bold")
sns.lineplot(data=trends);
```


# A Regression Model #

What makes time series unique is that consecutive observations in the series will usually be *dependent*. Today tells us about tomorrow.

So, since the time index informs us about the observations, we could treat a forecasting problem as a regression problem. We could treat the series of observations as the target and the index as a feature.

In this lesson, we will use a [simple linear regression ](https://en.wikipedia.org/wiki/Simple_linear_regression) model from the `statsmodels` module. `statsmodels` is like the `sklearn` of time series and other statistical models. We'll use it throughout this course.


```python
from sklearn.model_selection import train_test_split

data = trends.copy()
data['Week'] = range(len(data.Interest))
train_data, val_data = train_test_split(data, test_size = 0.2, shuffle = False)
```

The easiest way to get started with `statsmodels` is through its [formula interface](https://www.statsmodels.org/stable/example_formulas.html). Formulas in `statsmodels` work the same way as formulas in R. Instead of passing in our variables as arrays, we specify the regression relationship with a special kind of string and let `statsmodels` create the arrays for us. For OLS, the form is `"target ~ feature"`.

```python
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error

trends_model = smf.ols("Interest ~ Week", train_data).fit()
trends_model.summary()
```

Regression diagnostics.

```python
# In-sample predictions
train_predictions = trends_model.predict(train_data)
print(mean_absolute_error(train_data.Interest, train_predictions))
```

Now we'll make a forecast.

```python
# Out of sample predictions (the forecast)
val_predictions = trends_model.predict(val_data)
print(mean_absolute_error(val_data.Interest, val_predictions))
```


```python
plt.figure(figsize=(12,6))
plt.title("Fitted and forecast predictions from the regression model", fontweight="bold")
plt.xlabel("Date", fontweight="bold")
plt.ylabel("Interest", fontweight="bold")
plt.legend(title="Predictions", loc="upper left")
sns.lineplot(data=data.Interest, color="k")
sns.lineplot(data=train_predictions, label="Fitted", color="b")
sns.lineplot(data=val_predictions, label="Forecast", color="r");
```

## Your Turn ##

You learned how to do forecasts with linear regression. When you're ready, move on to the first exercise!





# Draft

Our least-squares model is equivalent to the *[Average method](https://otexts.com/fpp2/simple-methods.html)* of forecasting but applied to weekly changes in interest.


This temporal dependence is both a useful source of information, but also a strong constraint. Most of the methods you've used in previous courses ...


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
