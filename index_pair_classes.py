import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings as warnings
import scipy.stats as stats
import statsmodels.api as sm

NEGATIVE_SIGNAL = -1

def annualise(finalret, std, N):
    #return annualised
    avgret = 100 * (np.power((1+finalret) , 12 / N) - 1)
    #std annualised
    annualstd = float(std * np.sqrt(12.0))
    return avgret, annualstd

class SentimentIndex:
    def __init__(self,index_data):
        dates = index_data.index
        self.dates = dates
        self.start_date = dates[0]
        self.end_data = dates[-1]
        self.gsiindex = index_data*100
    
    def plotIndex(self):
        fig = plt.figure(figsize=(15,5))
        plt.plot(self.dates[:13],self.gsiindex[:13],'--b',label='Percentiles Pre 2009')
        plt.plot(self.dates[12:], self.gsiindex[12:], 'b', label = 'Percentiles 2009 Onwards')
        plt.xticks(self.dates[::int(len(self.dates)/15)])
        plt.title('Geopolitical Sentiment Index 2008-2023', fontsize=15)
        plt.xlabel('Dates', fontsize=11)
        plt.ylabel('Percentile')
        plt.legend()
        plt.show()
        
    def movingAverage(self, time_window):
        if time_window == 0:
            return self.gsiindex
        index_copy = self.gsiindex.copy()
        index_copy_MAs = index_copy['Percentile'].rolling(window=time_window).mean()
        
        return index_copy_MAs
        
    def percentageChange(self, time_window):
        if time_window == 0:
            return self.gsiindex
        index_copy = self.gsiindex.copy()
        index_copy_percs = index_copy['Percentile'].pct_change(time_window)
        
        return index_copy_percs
    

class TradePair:
    def __init__(self, pair_data):
        pair_data_copy = pair_data.copy()
        pair_data_copy.index = pd.to_datetime(pair_data_copy.index, format='%d/%m/%Y')
        pair_data_copy.index = pair_data_copy.index.to_series().apply(lambda x: x.strftime('%Y/%m'))
        self.pair_data = pair_data_copy
        self.dates = pair_data_copy.index
        self.start_date = self.dates[0]
        self.end_date = self.dates[-1]
    
    def equityCurve(self, plot=True):
        
        pair_values = self.pair_data['Value']
        pair_values /= 100
        pair_values += 1
        pair_values = pair_values.cumprod()
        if plot:
            fig = plt.figure(figsize=(15,5))
            if self.dates[0][:4] == '2008':
                plt.plot(self.dates[:13], pair_values[:13], '--b')
                plt.plot(self.dates[12:], pair_values[12:], color='blue', label = 'Equity Curve')
            else:
                plt.plot(self.dates, pair_values, 'blue', label='Equity Curve')
            plt.title('Equity Curve', fontsize=15)
            plt.xlabel('Dates',fontsize=11)
            plt.ylabel('Return',fontsize=11)
            plt.xticks(self.dates[::int(len(self.dates)/15)])
            plt.show()
        
        equity_df = pd.DataFrame(data=pair_values, index=self.dates)
        self.equity = equity_df


    def signalsMovingAverage(self, time_window, gsi_index, reverse=False):

        gsi_MAs = gsi_index.movingAverage(time_window = time_window)
        gsi_MAs = gsi_MAs.values[12:]
        gsi_dates = gsi_index.dates
        gsi = gsi_index.gsiindex.values[12:]
        signals = [0]

        for i in range(len(gsi_dates[12:])-1):
            if gsi_MAs[i] > gsi[i]:
                signal = NEGATIVE_SIGNAL if reverse else 1
                signals.append(signal)
            else:
                signal = 1 if reverse else NEGATIVE_SIGNAL
                signals.append(signal)
        
        return signals

    def tradePairMA(self, time_window, gsi_index, reverse=False):

        trade_signals = self.signalsMovingAverage(time_window=time_window, gsi_index=gsi_index, reverse=reverse)

        pair_vals = self.pair_data
        pair_vals = pair_vals['Value'][12:]

        total_equity = [1]
        traded_value = [0]

        for i in range(1,len(trade_signals)):
            trade = float(trade_signals[i]*pair_vals[i])
            traded_value.append(trade)
            total_equity.append(total_equity[i-1] * (1 + (trade/100)))

        final_return = total_equity[-1] - 1
        monthly_std = np.std(traded_value)
        N = len(total_equity)

        ret, std = annualise(final_return, monthly_std, N)
        sharpe_ratio = ret / std

        out = [np.round(ret, 4), np.round(std, 4),  np.round(sharpe_ratio, 4)]

        return out, total_equity
    
    def signalsPercentageChange(self, time_window, gsi_index, reverse=False):

        gsi_MAs = gsi_index.percentageChange(time_window = time_window)
        gsi_MAs = gsi_MAs.values[12:]
        gsi_dates = gsi_index.dates
        gsi = gsi_index.gsiindex.values[12:]
        signals = [0]

        for i in range(len(gsi_dates[12:])-1):
            if gsi_MAs[i] > 0:
                signal = NEGATIVE_SIGNAL if reverse else 1
                signals.append(signal)
            else:
                signal = 1 if reverse else NEGATIVE_SIGNAL
                signals.append(signal)
        
        return signals

    def tradePairPercs(self, time_window, gsi_index, reverse=False):

        trade_signals = self.signalsPercentageChange(time_window=time_window, gsi_index=gsi_index, reverse=reverse)

        pair_vals = self.pair_data
        pair_vals = pair_vals['Value'][12:]

        total_equity = [1]
        traded_value = [0]

        for i in range(1,len(trade_signals)):
            trade = float(trade_signals[i]*pair_vals[i])
            traded_value.append(trade)
            total_equity.append(total_equity[i-1] * (1 + (trade/100)))

        final_return = total_equity[-1] - 1
        monthly_std = np.std(traded_value)
        N = len(total_equity)

        ret, std = annualise(final_return, monthly_std, N)
        sharpe_ratio = ret / std

        out = [np.round(ret, 4), np.round(std, 4),  np.round(sharpe_ratio, 4)]

        return out, total_equity