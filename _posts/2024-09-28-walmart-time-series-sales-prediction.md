---
title: "(WIP) Time Series Forecast: Predicting Daily Sales for Walmart"
date: 2024-09-28
tags: [Time Series, Retail, Forecasting]
#header: ""
classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
header:
  image: /assets/img/timeSpace.jpg
  caption: "Image by [Gerd Altmann](https://pixabay.com/users/geralt-9301/) on [Pixabay](https://pixabay.com//)"
#image: timeSpace.jpg
excerpt: "Discover how I used time series-oriented machine learning models to forecast daily sales for Walmart stores, uncovering insights that could reshape retail decision-making"
#mathjax: "true"
---
# Problem exposition and Business Value

Walmart has Weekly Sales data spanning almost 3 years for 45 of its stores in the US. 

I have been tasked to come up with a method to forecast its sales. These predictions, if accurate and reliable, will be
extremely  useful to Walmart for several reasons:

1. *Operations & Inventory Optimization*: Predicting daily sales enables Walmart to streamline its operations by 
   optimizing inventory management. Accurate sales forecasts help ensure that the right products are stocked at the 
   right locations, reducing excess inventory and minimizing stockouts. Additionally, these predictions can help 
   Walmart optimize staffing levels, aligning workforce availability with customer traffic and demand, which ensures 
   that  stores are adequately staffed during peak times while minimizing labor costs during slower periods.

2. *Enhanced Customer Experience & Service*: Sales predictions are key to providing a smoother, more reliable customer 
   experience. By anticipating demand trends, Walmart can ensure that high-demand products are always available, 
   avoiding the frustration of out-of-stock items and thereby improving customer satisfaction. Additionally, this
   insight allows for more personalized promotions and marketing efforts, targeting customers with relevant offers
   based on anticipated purchasing trends.

3. *Supply Chain Efficiency & Cost Reduction*: Predicting daily sales helps Walmart manage its supply chain more
   effectively by reducing lead times and enabling just-in-time inventory practices. With better forecasts, the company
   can coordinate with suppliers to avoid last-minute rush orders, reducing shipping costs and improving product
   availability. Moreover, sales predictions support more accurate production scheduling for Walmart’s suppliers,
   fostering a more efficient and responsive supply chain.


# 1) Imports and set-up

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from scipy.stats import boxcox
from statsmodels.tsa.arima.model import ARIMA

plt.style.use('fivethirtyeight')
```

# 2) Load Dataset
```python
# load dataset

df = pd.read_csv('path_to_data/walmart-sales-dataset-of-45stores.csv')
```

# 3) EDA and Data Cleaning


Let's take a look at the data to understand it better

```python
# explore
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>Date</th>
      <th>Weekly_Sales</th>
      <th>Holiday_Flag</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>05-02-2010</td>
      <td>1643690.90</td>
      <td>0</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>211.096358</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>12-02-2010</td>
      <td>1641957.44</td>
      <td>1</td>
      <td>38.51</td>
      <td>2.548</td>
      <td>211.242170</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>19-02-2010</td>
      <td>1611968.17</td>
      <td>0</td>
      <td>39.93</td>
      <td>2.514</td>
      <td>211.289143</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>26-02-2010</td>
      <td>1409727.59</td>
      <td>0</td>
      <td>46.63</td>
      <td>2.561</td>
      <td>211.319643</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>05-03-2010</td>
      <td>1554806.68</td>
      <td>0</td>
      <td>46.50</td>
      <td>2.625</td>
      <td>211.350143</td>
      <td>8.106</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6430</th>
      <td>45</td>
      <td>28-09-2012</td>
      <td>713173.95</td>
      <td>0</td>
      <td>64.88</td>
      <td>3.997</td>
      <td>192.013558</td>
      <td>8.684</td>
    </tr>
    <tr>
      <th>6431</th>
      <td>45</td>
      <td>05-10-2012</td>
      <td>733455.07</td>
      <td>0</td>
      <td>64.89</td>
      <td>3.985</td>
      <td>192.170412</td>
      <td>8.667</td>
    </tr>
    <tr>
      <th>6432</th>
      <td>45</td>
      <td>12-10-2012</td>
      <td>734464.36</td>
      <td>0</td>
      <td>54.47</td>
      <td>4.000</td>
      <td>192.327265</td>
      <td>8.667</td>
    </tr>
    <tr>
      <th>6433</th>
      <td>45</td>
      <td>19-10-2012</td>
      <td>718125.53</td>
      <td>0</td>
      <td>56.47</td>
      <td>3.969</td>
      <td>192.330854</td>
      <td>8.667</td>
    </tr>
    <tr>
      <th>6434</th>
      <td>45</td>
      <td>26-10-2012</td>
      <td>760281.43</td>
      <td>0</td>
      <td>58.85</td>
      <td>3.882</td>
      <td>192.308899</td>
      <td>8.667</td>
    </tr>
  </tbody>
</table>
<p>6435 rows × 8 columns</p>
</div>


The data looks as expected. Let's call the .describe() method  to understand the distribution of the different columns
and see if there are any missing values

```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>Weekly_Sales</th>
      <th>Holiday_Flag</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6435.000000</td>
      <td>6.435000e+03</td>
      <td>6435.000000</td>
      <td>6435.000000</td>
      <td>6435.000000</td>
      <td>6435.000000</td>
      <td>6435.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.000000</td>
      <td>1.046965e+06</td>
      <td>0.069930</td>
      <td>60.663782</td>
      <td>3.358607</td>
      <td>171.578394</td>
      <td>7.999151</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.988182</td>
      <td>5.643666e+05</td>
      <td>0.255049</td>
      <td>18.444933</td>
      <td>0.459020</td>
      <td>39.356712</td>
      <td>1.875885</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.099862e+05</td>
      <td>0.000000</td>
      <td>-2.060000</td>
      <td>2.472000</td>
      <td>126.064000</td>
      <td>3.879000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.000000</td>
      <td>5.533501e+05</td>
      <td>0.000000</td>
      <td>47.460000</td>
      <td>2.933000</td>
      <td>131.735000</td>
      <td>6.891000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.000000</td>
      <td>9.607460e+05</td>
      <td>0.000000</td>
      <td>62.670000</td>
      <td>3.445000</td>
      <td>182.616521</td>
      <td>7.874000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>34.000000</td>
      <td>1.420159e+06</td>
      <td>0.000000</td>
      <td>74.940000</td>
      <td>3.735000</td>
      <td>212.743293</td>
      <td>8.622000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>45.000000</td>
      <td>3.818686e+06</td>
      <td>1.000000</td>
      <td>100.140000</td>
      <td>4.468000</td>
      <td>227.232807</td>
      <td>14.313000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    Store           0
    Date            0
    Weekly_Sales    0
    Holiday_Flag    0
    Temperature     0
    Fuel_Price      0
    CPI             0
    Unemployment    0
    dtype: int64




```python
print(df.Date.min())
print(df.Date.max())
```

    01-04-2011
    31-12-2010



```python
df.Date.value_counts()
```




    05-02-2010    45
    23-12-2011    45
    11-11-2011    45
    18-11-2011    45
    25-11-2011    45
    02-12-2011    45
    09-12-2011    45
    16-12-2011    45
    30-12-2011    45
    28-10-2011    45
    06-01-2012    45
    13-01-2012    45
    20-01-2012    45
    .
    .
    .
    31-12-2010    45
    07-01-2011    45
    14-01-2011    45
    21-01-2011    45
    28-01-2011    45
    26-10-2012    45
    Name: Date, dtype: int64



Everything looks to be in order. There are no missing values. Each store has 45 data points, as does each date.
Something to note is that standard deviation of the Sales metric is quite high.

# 4) Problem Framing

We need to do a little work to better frame the problem. I had originally wanted to use a single store to predict 
weekly sales for, but this number of data points is quite small (45).

The alternative is to combine several stores data at the date level by summing weekly sales. This will solve the
problem of not having enough data, but it will probably introduce spurious variables into the dataset.

In other words, there could be differences inherent to the stores I would be combining, such as one store being located 
in the Downtown part of a major city while the other is in a more rural location, which could impact their sales.
I won't be able to capture this in my model and this will likely introduce errors.

One way to overcome this problem is to combine data for stores that exhibit similar levels of sales, so let's try
doing just that


```python
df.sort_values(by = ['Store', 'Date'], inplace = True)
sales_line_plot_df = df[['Store', 'Date', 'Weekly_Sales']]

sales_line_plot_df = sales_line_plot_df.groupby(['Store', 'Date'])['Weekly_Sales'].max()
sales_line_plot_df = sales_line_plot_df.unstack(level = ['Store'])

# take the first 20 stores as a sample
sales_line_plot_df_sample_20 = sales_line_plot_df.iloc[:, :20]
sales_line_plot_df_sample_20.plot(figsize = (12, 10))
```



<img src="{{ site.url }}{{ site.baseurl }}/images/walmart-time-series/output_12_1.png" alt="None">
    


A very messy graph but it serves our purpose - we can clearly see that there are clusters of stores with similar 
volumes of Weekly Sales. It looks like Stores 8, 12, 17 and 18 are clustered together. Let's see if this is indeed
the case. (I also looked at the stores 21 to 40 and added these to the below graph as they have a similar
volume of sales)


```python
sales_line_plot_df.loc[:, [8, 12, 17, 18, 35, 40]].plot(figsize = (12, 10))
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_14_1.png)
    

Nice. Let's go ahead and create a new dataframe with only our require stores. Let's also combine the sales of all 
stores at each given Date, the values of which are thankfully consistent across stores

```python
df_selected_stores = df[df.Store.isin([8, 12, 17, 18, 35, 40])]
df_selected_stores = df_selected_stores.groupby('Date')['Weekly_Sales'].sum().to_frame()

# set the date type of the dataframe to datetime for further processing and sort it
df_selected_stores.index = pd.to_datetime(df_selected_stores.index)
df_selected_stores.sort_index(inplace = True)
```

Let's do a final check on the combined weekly data for our 5 stores
```python


df_selected_stores.plot(kind = 'line', figsize = (10, 8))
```

    
![png](output_16_1.png)
    


Alright, before we do a persistence baseline evaluation, we need to resample the data. This is because the Date 
values are erratic and don't follow any real pattern (see below)


```python
pd.set_option('display.max_rows', 400)

pd.Series(df_selected_stores.index.unique())
```




    0     2010-01-10
    1     2010-02-04
    2     2010-02-07
    3     2010-02-19
    4     2010-02-26
    5     2010-03-09
    6     2010-03-12
    7     2010-03-19
    8     2010-03-26
    9     2010-04-06
    10    2010-04-16
    11    2010-04-23
    12    2010-04-30
    13    2010-05-02
    .
    .
    .
    138   2012-10-08
    139   2012-10-19
    140   2012-10-26
    141   2012-11-05
    142   2012-12-10
    Name: Date, dtype: datetime64[ns]



It seems reasonable that a supermarkets sales follow a daily cycle - many customers shop on the weekend, for example. 
Alternatively we can also create a slice of the data that displays weekly or monthly data.

Initially I wanted to have a weekly view. However, the Date values are irregular and this is problematic; in many cases,
a given week has only one data point, while other weeks have 0 data points and others still have multiple data points.
Downsampling to weekly data in this case gives us NaNs for many weeks.

As a result, I will upsample to Daily data and interpolate views between existing data points using a quadratic 
function. If our model's performance is poor on this view, we can always try a monthly framing of the problem.


```python
upsampled_df_selected_stores = df_selected_stores.resample('D')
upsampled_df_selected_stores = upsampled_df_selected_stores.interpolate(method = 'spline', order = 2)

# rename column to reflect new granularity
upsampled_df_selected_stores.rename({'Weekly_Sales':'Daily Sales'}, axis = 1, inplace = True)
```


```python
upsampled_df_selected_stores.plot()
```

    
![png](output_21_1.png)
    


Before we do anything else, we need to have a baseline performance to compare our eventual model and predictions to. In time series problem, a persistence or naive forecast is often used for this purpose. Put simply, we will use the value at obs(t-1) as a prediction for the value at obs(t). 

To do this, we split the data into a training and test set, setting aside 90% of the data for the training set. The test size is small because we will later need to iterate through our ARIMA model len(test) times. In other words, if len(test) is large, it will take a long time for our ARIMA model to fit

We will then walk forward over the test set, adding each 'new' observation seen in the test set to our training set, and using that as the prediction for the next time stamp


```python

# split into train and test sets

X = upsampled_df_selected_stores['Daily Sales'].values
train_size = int(len(X)*0.90)

train, test = X[0:train_size], X[train_size:]

history = [x for x in train]
predictions = []

for i in range(len(test)):
    yhat = history[-1] # the very last observation seen in 'history'
    predictions.append(yhat)
    latest_observation = test[i]  # add the value we have just seen to 'history'
    history.append(latest_observation)

    
# find RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print('RMSE for Persistence Model is £%.2f in Sales per Day' % rmse)
```

    RMSE for Persistence Model is £45872.02 in Sales per Day



```python
for i in range(20):
    print('Predicted Value: £%.d | Actual Value: £%d' % (predictions[i], test[i]))
```

    Predicted Value: £5927130 | Actual Value: £5953795
    Predicted Value: £5953795 | Actual Value: £5975201
    Predicted Value: £5975201 | Actual Value: £5989949
    Predicted Value: £5989949 | Actual Value: £5989640
    Predicted Value: £5989640 | Actual Value: £5972872
    Predicted Value: £5972872 | Actual Value: £5939646
    Predicted Value: £5939646 | Actual Value: £5889961
    Predicted Value: £5889961 | Actual Value: £5826909
    Predicted Value: £5826909 | Actual Value: £5769028
    Predicted Value: £5769028 | Actual Value: £5719410
    Predicted Value: £5719410 | Actual Value: £5678053
    Predicted Value: £5678053 | Actual Value: £5644958
    Predicted Value: £5644958 | Actual Value: £5620126
    Predicted Value: £5620126 | Actual Value: £5603555
    Predicted Value: £5603555 | Actual Value: £5593979
    Predicted Value: £5593979 | Actual Value: £5583791
    Predicted Value: £5583791 | Actual Value: £5571725
    Predicted Value: £5571725 | Actual Value: £5557781
    Predicted Value: £5557781 | Actual Value: £5541958
    Predicted Value: £5541958 | Actual Value: £5524256


The persistence forecast error is only about £46k per day. As a % of the mean sales over the time period, we see that it is only around a 0.8% error

Still, this is the target to beat for our model and predictions


```python
rmse / np.mean(X)
```




    0.007974285431459617



Let's look at the data more closely


```python
plt.figure(figsize = (12, 6))
plt.plot(upsampled_df_selected_stores)
```




    [<matplotlib.lines.Line2D at 0x7fd461da1510>]




    
![png](output_28_1.png)
    


- At least 4 big outlier values, all of which occur towards the end of the year
- Seems like there is some seasonal/cyclical behaviour, but it is erratic
- There does not seem to be a trend in the data

Let's check if the data is stationary using the Augmented Dickey-Fuller test. The null hypothesis for this test is that the series has unit root, and therefore is non-stationary

We see below that the test statistic is very extreme and the p-value is 0. Hence, we reject the null hypothesis - this series is stationary


```python
result = adfuller(X)
print('ADF Test Statistic: %.3f' % result[0])
print('p-value: %.3f' % result[1])
```

    ADF Test Statistic: -5.626
    p-value: 0.000


Manually configured ARIMA

Let's manually configure an ARIMA model using Autocorrelation and Partial Autocorrelation Plots

From the below graphs, we see that correlations with up to 7 lagged values are significant. So we can start with value of 7 for the 'p' or Autoregression parameter

Additionally, the PACF suggests we can use a value of 4 for the 'q' or Moving Average parameter


```python
plot_acf(upsampled_df_selected_stores);
```


    
![png](output_32_0.png)
    



```python
plot_pacf(upsampled_df_selected_stores, lags = 15, method = 'ywm');
```


    
![png](output_33_0.png)
    


To fit and evaluate our ARIMA model, we need to execute the same method of walk forward validation as we did with the persistence model. That is to say, we will:

- train the model on the entire training set
- make a 1-step prediction
- add the 'current' value of the test set to the training set
- retrain the model on the training and repeat, until we have come to the end of the test set

Through trial and error, I have found that large ARIMA values (like 7) result in a Convergence Error. If we reduce the number of lagged values to be included, we overcome this error


```python
history = [x for x in train]
predictions = []

for i in range(len(test)):
    model = ARIMA(history, order = (3, 0, 4))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    latest_obs = test[i]
    history.append(latest_obs)
```


```python
rmse = np.sqrt(mean_squared_error(test, predictions))
print('RMSE for ARIMA Model is £%.2f in Sales per Day' % rmse)
```

    RMSE for Persistence Model is £7289.27 in Sales per Day



```python
for i in range(20):
    print('Predicted Value: £%.d | Actual Value: £%d' % (predictions[i], test[i]))
```

    Predicted Value: £5950185 | Actual Value: £5953795
    Predicted Value: £5972777 | Actual Value: £5975201
    Predicted Value: £5987227 | Actual Value: £5989949
    Predicted Value: £5995443 | Actual Value: £5989640
    Predicted Value: £5973543 | Actual Value: £5972872
    Predicted Value: £5944142 | Actual Value: £5939646
    Predicted Value: £5895898 | Actual Value: £5889961
    Predicted Value: £5832052 | Actual Value: £5826909
    Predicted Value: £5761621 | Actual Value: £5769028
    Predicted Value: £5724390 | Actual Value: £5719410
    Predicted Value: £5679737 | Actual Value: £5678053
    Predicted Value: £5647597 | Actual Value: £5644958
    Predicted Value: £5622036 | Actual Value: £5620126
    Predicted Value: £5603700 | Actual Value: £5603555
    Predicted Value: £5596953 | Actual Value: £5593979
    Predicted Value: £5591627 | Actual Value: £5583791
    Predicted Value: £5574931 | Actual Value: £5571725
    Predicted Value: £5562750 | Actual Value: £5557781
    Predicted Value: £5546685 | Actual Value: £5541958
    Predicted Value: £5529952 | Actual Value: £5524256


Wow, we were able to drastically reduce the RMSE compared to the persistence forecast model. In fact, it is a (7289.27 - 45872.02)/45872.02 = 84% reduction in RMSE

However, we might be able to do even better. Above, I manually configured the ARIMA model to be used. We can now take a more analytical approach and perform Grid Search to find the optimal values for p, d and q

To do this, I need to wrap my ARIMA code in a function. I then need to define another function that performs grid search, and within this seconf function, I need to call my ARIMA function


```python
import warnings

# define ARIMA evaluation function to be called in evaluate_models function
def evaluate_arima_model(X, arima_order): 
    
    train_size = int(len(X) * 0.90)
    train, test = X[0:train_size], X[train_size:]
    
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

# define evaluate_models function - this is the gridsearch function

def evaluate_models(dataset, p_val, d_val, q_val):
    best_score, best_config = float("inf"), None
    for p in p_val:
        for d in d_val:
            for q in q_val:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order) #dataset has to be a Series
                    if rmse < best_score:
                        best_score, best_config = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE = %.3f' % (best_config, best_score))
    
# to call these functions, first define the ranges of p, d and q you want to iterate through

p_values = range(0, 6)
d_values = range(0, 1)
q_values = range(0, 12)
warnings.filterwarnings("ignore")

# call one function inside the other.
evaluate_models(X, p_values, d_values, q_values)
```


```python

```
