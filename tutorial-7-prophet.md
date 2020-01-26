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

# Notes #

```python
import pandas as pd
from statsmodels.tsa.tsatools import detrend
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
pd.plotting.register_matplotlib_converters()

df = pd.read_csv("data/datascience.csv")
df.rename(columns={"Week":"ds", "Interest":"y"}, inplace=True)
df["y"] = detrend(df.y)/4
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

sns.set_style("white")
m.plot(forecast)
sns.axes_style({'axes.grid'})
plt.show()

m.plot_components(forecast)
plt.show()
```
