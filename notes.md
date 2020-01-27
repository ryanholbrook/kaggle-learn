
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

```python
sns.lineplot(data=sm.tsa.tsatools.detrend(trends.Interest), color="b")
# sns.lineplot(data=(train_data.Interest - train_predictions), color="r")
# sns.lineplot(data=(val_data.Interest - val_predictions), color="r")
sns.lineplot(data=(trends.Interest.diff()), color="g")
plt.show()
```

```python
import numpy.random
import statsmodels.api as sm
tmp = pd.DataFrame(numpy.random.normal(0, 40, 250) + range(1, 251))
plt.figure(figsize=(20,4))
ax = sns.lineplot(data=tmp)
sm.graphics.abline_plot(intercept=1, slope=1, ax=ax, color='r', lw=1)
plt.gca().legend().remove()
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.savefig("linear-trend.png")
plt.show()
```

## Looking Ahead ##

Sometimes, you'll want to model the *change* in a time series instead of the values of the observations themselves. In finance, for instance, it is much more common to model the *returns* of a stock than to model its price.

It turns out that our least-squares model is equivalent (for infintely-large samples) to a simple baseline model known as the [average method](https://otexts.com/fpp2/simple-methods.html), when the average method is used on the *change* in values over each period. The average method would predict that all future change in the popularity of "data science" would be the average of all the observed change. As we saw, #TODO# our model predicted change each week #END_TODO#.

We will look at a number of powerful time-series models in this micro-course. Nonetheless, it is important to understand even simplest models like the average method. There's no reason to waste time on a complicated model if it can't even beat a simple average!
