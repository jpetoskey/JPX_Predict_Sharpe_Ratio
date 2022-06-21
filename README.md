# Capstone Project - JPX Tokyo Stock Exchange Prediction

## Overview

This is a time-series modelling competition to produce predictions for the Sharpe Ratio for 2000 stocks once each day for 56 days.  

The contestant's one-day score is computed two days following the submission of the Sharpe Ratio and is based on the return of the user's top 200 and bottom 200 stocks.  The top 200 stocks are bought and the bottom 200 are shorted on the close of the business day after the Sharpe Ratio is predicted and sold at the close of the following trading day.  The mean of the 56 single day scores is calculated, then divided by the standard deviation of those scores for the final competition score.

   * Data from: [Kaggle Competition: JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)  

   * [Overview for the JPX Tokyo Stock Exchange on Kaggle](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)
  

# Business Opportunity

JPX wants to increase the number of trades that are made each day, so they have devised a competition to help people feel confidence in trading on a daily basis.

For an individual investor, there may be a business opportunity.  However, for the JPX Stock Exchange, there is a huge benefit to providing institutional investors with models that generate daily trading returns as this could dramatically increase the volume of trades, and thereby revenue.

This competition seems likely to yield a number of models that make money trading on a daily basis, at scale.


# Data
   
Same link as above, but relevant for perusing data: [Kaggle Competition: JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)

Predictions need to be made based on 2000 stocks, however data from additional stocks are provided to support modelling.

## The data is broken down into 4 main categories: Prices, Financials, Stock List, Options:

Prices data includes daily trading values such as Open, Close, Volume, with some additional attributes, such as Expected Dividend and Adjustment Factor.

Financials data includes income statement information and is usually posted quarterly for each security.

Stock List data contains categorical information, such as market sector, and some numerical information, such as market capitalization.

Options data includes options trading information and has a large amount of trading-focused data.


# Methods

In terms of modeling, Time-Series Models such as ARIMA and Prophet were implemented initially, then a LSTM Neural Network with a Dense Layer, then a Random Forest Regressor.  
   
## ARIMA and Prophet
[ARIMA Final Notebook](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/code/2.%20ARIMA/ARIMA%20Final.ipynb)
[Prophet Notebook](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/code/3.%20Prophet/FB%20Prophet.ipynb)
Time series models, such as ARIMA and Prophet largely predicted near the mean as the target variable passes the stationarity test for every stock tested.  Predicting near the securities' mean did not yield a high score on kaggle as it was not able to accurately differentiate between stocks' Sharpe Ratio on a daily basis.
       
The ARIMA and Prophet models relied solely on how the Sharpe Ratio changed over time and did not have access to the variety of data available to regressors or neural networks.  Considering there were no trends or seasonality to glean from the target variable, these models did not have much valuable information to decrease error or improve differentiation between stocks. I did attempt to add a regressor to the prophet models, but was unable to make the add regressor function work and moved on to trying a neural network.
  
## LSTM Neural Network
[LSTM Neural Network Final Notebook](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/code/5.%20LSTM/LSTM%20Final.ipynb)
Moving on to Neural Networks, via an LSTM model with a Dense layer yielded lower error, but similar results of predicting near the mean for each stock and not being able to differentiate between the Sharpe Ratio on a daily basis.

The LSTM model did have access to more data, but it was not clear to me why it wasn't able to make predictions further from the mean.  I would need to do more learning and research to better understand why it prioritized decreasing error in this manner.

## Random Forest Regressor
[LSTM Neural Network Final Notebook](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/code/4.%20Random_Forest_Regressor/Random%20Forest%20Regressor%20Final.ipynb)
Lastly, Random Forest Regressor Models performed better at differentiating between stocks on a daily basis, though they did not perform much better in terms of Root Mean Squared Error for most stocks.  One of the challenges with Random Forest Regressors is properly training them on a data frame that allows them to interpret date, and this was accomplished by converting the date to a float and including it in the training set. 

In addition to including the date in the training set, I calculated a Sharpe Ratio using the close of the prior and current day, which would be part of the test set, or prediction set, on Kaggle.  The calculation was inspired by an article on [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm) by Stanford's William F. Sharpe himself.  This feature addition contributed 26% of the feature importance, of 29 features in total, with the second most important feature being the 2 Day Spread at 7%.
   
Two methods of modelling were used in the Random Forest Regressors, and the second type of modelling - creating a separate model and prediction for each of 2000 stocks did much better at differentiating between stocks than the single model, trained on all of the stocks at once.


# Results

All models types were able to achieve a relatively similar average Root Mean Squared Error of somewhere around 0.020, but each model type performed quite differently in terms of scoring, and it is apparent why this occurred when observing plots for a given stock.

## ARIMA and Prophet
ARIMA models scored below 0 in most cases, with my final test scoring a -1.43 on Kaggle.  The reason for this seems to be because it chose to predict very close to the mean target value for a given stock and wasn't able to determine when the target would rise or fall.  This makes sense, because the target values pass a stationarity check and there are no trends or seasonality for the ARIMA model to recognize.

I did not complete a final submission to score for a Prophet model as it was performing very similarly to ARIMA and I wanted to move on to modelling with neural networks.

### Visual 1
[ARIMA Model for Stock 9663](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/images/ARIMA_9663.png)

## LSTM Neural Network with Dense Layer
The LSTM model was not able to produce results for every stock even when implemented on Google Cloud with 4 CPUs, 16 GB of RAM and 1 GPU.  However, it seems likely that the model would have scored similarly to the ARIMA, as it was predicting target values very close to the mean.

### Visual 2
[LSTM with Dense Layer for Stock 9663](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/images/LSTM_9663.png)

## Random Forest Regressors
The Random Forest Regressors were more successful at differentiating between stocks' Sharpe Ratio and this resulted in much higher Kaggle scores.  The predictions ranged much farther from the mean Sharpe Ratio value and were able to attain a similar or better RMSE compared to the ARIMA and LSTM models.

In the visuals below, note how the individual random forest model is able to reduce the error on its predictions quite dramatically when stock 9663 experiences volatility.  The Single Random Forest Model is trained on all stocks at once and the individual random forest model is trained on each stock independently.

### Visual 3
[Single Random Forest Model for Stock 9663](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/images/RFR_1Model_9663.png)

### Visual 4
[Individual Random Forest Model for Stock 9663](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/images/RFR_IndivModel_9663.png)

## Comparing RMSE between Models for 5 Stocks
Note how a given model type does not always have the best RMSE, but the Individually Modelled Random Forest Regressor does perform the best overall.  And, it performs much better on stocks that exhibit greater volatility, such as stock 9663.

### Visual 5
[Error by Model Type](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/images/Error%20by%20Model%20Type.png)

## Top Kaggle Score by Model Type
The top Kaggle Score by a wide margin was attained by the Individually Modelled Random Forest Regressor.  This seems due to its ability to make predictions that differentiate between stocks' Sharpe Ratio on a daily basis.

### Visual 6 
[Kaggle Scores](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/images/Kaggle%20Score%20by%20Model%20Type.png)


# Conclusion

## Generate Positive Returns
It seems likely that the Individually Modelled Random Forest Regressor will generate returns on a daily basis when implemented at a large scale, which involves ranking 2000 stocks, buying the top 200, and shorting the bottom 200 and repeating this process on a daily basis.  It seems less likely that this approach would work for individual or retail investors, as trading 400 stocks on a daily basis is not something I know how to do, but it could reap large returns for institutional investors, and most relevant to this competition, JPX Stock Exchange.


# Future Work Highly Recommended

## Add Options Data
I was able to add financial and stock list data into the training data frame, and the next step is adding Options data.  This trading information available in the options data set may give the Random Forest and Neural Network models more insight into what is going to happen in the market.

## Additional Regressor Model Types
Random Forest has done the best job predicting the Sharpe Ratio so far, but I am eager to try Naive Bayes as this model has done a better job making sense of stock market information in [other projects](https://datascience2.medium.com/comparing-every-machine-learning-algorithm-to-predict-the-stock-market-f2ddc06a0d91). 

# For More Information

   Please review my final analysis in my [Model Comparison Notebook](https://github.com/jpetoskey/JPX_Predict_Sharpe_Ratio/blob/main/Model%20Comparison.ipynb) or my [presentation]().

   For any additional questions, please contact **Jim Petoskey - Jim.Petoskey.146@gmail.com**

## Repository Structure

```
├── README.md                            <- The top-level README for reviewers of this project
├── Model Comparison.ipynb               <- Narrative documentation of analysis in Jupyter notebook
├── data                                 <- Contains notebooks for Data Exploration, ARIMA, Prophet, LSTM, and Random Forest Regressors
├── presentation                         <- Contains pdf, ppt, and recording of presentation directed towards JPX Stock Exchange                
└── images                               <- Generated from code with matplotlib and seaborn
```

## References

I utilized many resources over the course of this project and some especially helpful sources are included below:

- [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)

- Hastie, Tibshirani, and Friedman. 'The Elements of Statistical Learning'. Second Edition. Springer. 2009

- [ARIMA vs Prophet vs LSTM for Time Series Prediction](https://neptune.ai/blog/arima-vs-prophet-vs-lstm)

- [Comparing every machine learning algorithm to predict the stock market](https://datascience2.medium.com/comparing-every-machine-learning-algorithm-to-predict-the-stock-market-f2ddc06a0d91)

- [Random Forest for Time Series Forecasting](https://machinelearningmastery.com/random-forest-for-time-series-forecasting/)

- [Using CNN for financial time series prediction](https://machinelearningmastery.com/using-cnn-for-financial-time-series-prediction/)

- [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)

- [Prophet Diagnostics](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)
