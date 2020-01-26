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

You'll be prepared for this micro-course if you know how to [construct a machine learning model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model), [manipulate dataframes with Pandas](https://www.kaggle.com/residentmario/indexing-selecting-assigning), and [use seaborn to explore your data](https://www.kaggle.com/alexisbcook/hello-seaborn). You'll have a leg up if you've done some work on the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition, but we'll review what we need as we go.


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

You can quickly plot a data frame with the `plot` method. For time series, `pandas` will create a line chart by default.

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


# A Regression Model #

What makes a time series unique is that consecutive observations in the series will usually be *dependent*. Today tells us about tomorrow.

So, since the time index informs us about the observations, we could treat a forecasting problem as a regression problem. We could treat the series of observations as the target and the index as a feature.

To make a forecast on our series, we will fit a *linear trendline* using [simple linear regression ](https://en.wikipedia.org/wiki/Simple_linear_regression).

## Prepare Data ##


```python
from sklearn.model_selection import train_test_split

data = trends.copy()
# Add a numeric column for the model to fit on
data['Week'] = range(len(data.Interest))

# Split the data into a training set and a validation set
# The order of the observations is important, so don't shuffle
train_data, val_data = train_test_split(data, test_size = 0.2, shuffle = False)
```

## Define and Fit Model ##

`statsmodels` is like the `sklearn` of time series and other statistical models. We'll use it throughout this micro-course.

Let's take a moment to get acquainted with `statsmodels`.

The easiest way to get started with `statsmodels` is through its [formula interface](https://www.statsmodels.org/stable/example_formulas.html). Formulas in `statsmodels` work the same way as formulas in R. Instead of passing in our variables as arrays (like in `sklearn`), in `statsmodels` we can specify the regression relationship with a special kind of string and let `statsmodels` create the arrays for us.

For OLS, the form is `"target ~ feature"`.

```python
import statsmodels.formula.api as smf

# Fit an ordinary least-squares model using the formula interface
trends_model = smf.ols("Interest ~ Week", train_data).fit()
# Look at the fitted coefficients of the least-squares line
trends_model.params
```

The coefficients say: "Predict about 27 points of Interest for the first week, and about 0.24 more points for every week that goes by."


## Evaluate ##

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

### Interpret ###

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

There is an obvious seasonal ##cyclic?## component to our data. You can see that the popularity of "data science" tends to fall in the summer and winter and rise Spring and Fall. (Students on break from school?) We'll develop models in future lessons that can make use of information like this and give us better predictions.


## Conclusion ##

The defining feature of time series is their dependence on a temporal order. This temporal dependence is both a useful source of information, but also a strong constraint. If you haven't worked with time series before, it's likely that . The models you've proabaly worked with before function best, work best when your applied to data that are iid.


## Your Turn ##

So now you know how to make forecasts using a linear trendline. When you're ready, move on to the first exercise!


# Exercises #

## Load the Data ##

In this exercise, you'll investigate the popularity trend of the search term ["machine learning"](https://trends.google.com/trends/explore?date=2015-01-25%202020-01-25&geo=US&q=machine%20learning) as given by Google Trends.

## Plot It ##

## Split the Data ##

It is important that you do not shuffle time series data before splitting it. With time series, the order of our data sets must be preserved.

## Fit the Regression Model ##

## Evaluate ##

## Discuss ##



# Draft

Our least-squares model is almost equivalent to the *[Average method](https://otexts.com/fpp2/simple-methods.html)* of forecasting but applied to weekly changes in Interest. The expectation of both models are the same (the slope), but the error terms are different. OLS give the error term, but differencing gives a series of error terms.

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

### Looking Ahead ###

Sometimes, you'll want to model the *change* in a time series instead of the values of the observations themselves. In finance, for instance, it is much more common to model the *returns* of a stock than to model its price.

It turns out that our least-squares model is equivalent (for infintely large samples) to a simple baseline model known as the [average method](https://otexts.com/fpp2/simple-methods.html), when the average method is used on the *change* in values over each period. The average method would predict that all future change in the popularity of "data science" would be the average of all the observed change. As we saw, #TODO# our model predicted change each week #END_TODO#.

In the next lesson, you'll learn how to 

We will look at a number of powerful time-series models in this microcourse. Nonetheless, it is important to understand even simplest models like the average method. There's no reason to waste time on a complicated model if it can't even beat a simple average!

```R
1
```
