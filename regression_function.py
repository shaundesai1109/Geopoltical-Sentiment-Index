import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings as warnings
import scipy.stats as stats
import statsmodels.api as sm

from index_pair_classes import *

warnings.filterwarnings('ignore')

def regressionPair(gsi_index_data, pair_data, time_window, MA = True, plot=True):
    GSIObj = SentimentIndex(gsi_index_data)
    PairObj = TradePair(pair_data)
    
    y = PairObj.pair_data['Value'][13:-1]
    if MA:
        x = GSIObj.movingAverage(time_window).values[12:-2]
    else:
        x = GSIObj.percentageChange(time_window).values[12:-2]
    
    # Add constant term to the independent variable
    x = sm.add_constant(x)

    # Fit the linear regression model
    model = sm.OLS(y, x)
    results = model.fit()

    # Extract coefficients and their standard errors
    regression_coeffs = results.params
    std_errors = results.bse

    # T-statistics
    t_stat = regression_coeffs / std_errors

    # P-values
    p_values = results.pvalues

    if plot:
        fig = plt.figure(figsize=(10, 3))
        plt.scatter(x[:, 1], y, color='green', label='Actual Values')
        xfit = np.linspace(x[:, 1].min(), x[:, 1].max(), 1000)
        yfit = results.predict(sm.add_constant(xfit))
        plt.plot(xfit, yfit, color='orange', label='Fitted Values')
        plt.legend()
        if time_window > 0:
            plt.title(f'{time_window} Month Regression')
        else:
            plt.title(f'Standard Index Regression')
        plt.xlabel('Index Percentile')
        plt.ylabel('Pair Value')
        plt.show()

    return regression_coeffs[0], regression_coeffs[1], t_stat[0], t_stat[1], p_values[0], p_values[1]
