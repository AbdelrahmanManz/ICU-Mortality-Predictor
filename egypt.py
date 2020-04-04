import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py

def imputate():
    confirmed = pd.read_csv("global_data.csv", error_bad_lines=False)
    egypt_confirmed = confirmed.loc[confirmed['Country/Region']=="Egypt"]
    return egypt_confirmed

def create_ts(df):
    ts = df
    ts = ts.drop(['Province/State', 'Country/Region','Lat', 'Long'], axis=1)
    ts = ts.T
    ts = ts.fillna(0)
    ts = ts.reindex(sorted(ts.columns), axis=1)
    return ts


def run():
    egypt_confirmed = imputate()
    ts = create_ts(egypt_confirmed)
    p = ts.reindex(ts.max().sort_values(ascending=False).index, axis=1)
    p = p.iloc[25:, :1]
    # p.plot(marker='*', figsize=(10, 4)).set_title('Daily Total Confirmed - Egypt',
                                                              # fontdict={'fontsize': 22})
    # plt.show()

    prophecy = pd.DataFrame({'ds': list(p.index),
                             'y': [i[0] for i in p.values], })
    prophecy.columns = ["ds", "y"]
    prophecy["ds"] = pd.to_datetime(prophecy["ds"])
    m = Prophet(interval_width=0.95, growth='logistic')
    prophecy['cap']=300000
    prophecy['floor']=0
    m.fit(prophecy)
    future = m.make_future_dataframe(periods=1)
    future['cap'] = 300000
    future['floor'] = 0
    forecast = m.predict(future)
    print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
    fig = plot_plotly(m, forecast,plot_cap=False)  # This returns a plotly Figure
    py.iplot(fig)

run()
