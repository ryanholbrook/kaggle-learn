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
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
pd.plotting.register_matplotlib_converters()

df = pd.read_csv("data/machinelearning.csv")
df.rename(columns={"Week":"ds", "Interest":"y"}, inplace=True)
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast)
add_changepoints_to_plot(fig.gca(), m, forecast, cp_color="r", trend=True)
plt.title("Prophet", fontweight="bold")
plt.xlabel("Time", fontweight="bold")
plt.ylabel("Popularity", fontweight="bold")
plt.show()

m.plot_components(forecast)
plt.show()
```
