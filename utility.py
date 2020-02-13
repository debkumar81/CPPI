import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize

def price_return(price_series, window = 1, ret_type = "S"):
    """
    Takes a time series of asset prices
    Based on "ret_type", calculates either simple return or log-return
    Default is simple return
    "window" defines lookback window - eg, for daily prices, window = 1 means daily return, 5 means weekly return etc.
    Default "window" is 1
    """
    if (ret_type == "S"):
        return (price_series.pct_change(window)).dropna()
    elif (ret_type == "L"):
        return (log(price_series) - log(price_series.shift(window))).dropna()
    else:
        raise TypeError("Espected r to be Series or DF")


def compound(r):
    """
    returns the result of compounding the set of returns in r
    cumprod(1+r) = exp(sum(log(1+r)))-1
    """
    return np.expm1(np.log1p(r).sum())

def avg_return(r,  periods_per_year, annualize = True):
    """
    Calculates Average Return
    Annualiztion option by default True 
    Periods per Year is a function of whether the data is daily/weekly/monthly/quarterly/annual
    """
    n = r.shape[0]
    compounded_growth = (1 + r).prod()
    if annualize:
        return compounded_growth**(periods_per_year/n) - 1
    else:
        return compounded_growth**(1/n) - 1


def annualized_vol(r, periods_per_year):
    """
    Calculates Annualized Volatility
    DF = n-1
    Periods per Year is a function of whether the data is daily/weekly/monthly/quarterly/annual
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, periods_per_year, rf):
    """
    Calculates Annualized Sharp Ratio
    Periods per Year is a function of whether the data is daily/weekly/monthly/quarterly/annual
    rf : Risk Free rate
    """
    # convert the annualized risk free to risk free per period
    rf_per_period = (1+rf)**(1/periods_per_year) -1
    excess_ret = r - rf_per_period
    return avg_return(excess_ret, annualize = True, periods_per_year = periods_per_year)/annualized_vol(r, periods_per_year = periods_per_year)

def drawdown(return_series: pd.Series):
    """"
    Takes a time sries of asset return
    Creates Wealth Index, previous peaks and percent drawdowns
    """
    wealth_index = 100*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = 100*(wealth_index/previous_peaks - 1)
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdowns
    })

def var_h(r, level = 5):
    """
    Historic VaR
    r : DF or Series
    level : Level
    Returns Historic VaR
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_h, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Espected r to be Series or DF")

def var_gaussian(r, level = 5, modified = False):
    """
    Gaussian VaR
    r : DF or Series
    level : Level
    Returns Gaussian VaR
    If "modified" is True, then it uses Cornish-Fisher Modification to calculate Semi-parametric VaR
    """
    z = st.norm.ppf(level/100)
    if modified:
        # Modify Z-score based on Skewness and Kurtosis of the data
        s = skewness(r)
        k = kurtosis(r)
        z = ( z +
             (z**2 -1)*s/6 +
             (z**3 -3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof = 0))

def run_cppi(risky_r, safe_r = None, m = 3, initW = 1000, floor = 0.8, rf = 0.05, drawdown = None):
    """
    
    """
    # Set-up CPPI Parameters
    dates = risky_r.index
    account_value = initW
    peak = initW
    floor_value = initW*floor
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ["R"])
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = rf/12
                  
    account_hist = pd.DataFrame().reindex_like(risky_r)
    floor_hist = pd.DataFrame().reindex_like(risky_r)
    cushion_hist = pd.DataFrame().reindex_like(risky_r)
    risky_w_hist = pd.DataFrame().reindex_like(risky_r)   
    
    for step in range(len(dates)):
        if drawdown is not None:
            # Floor Value changes based on Peak and Drawdown
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1 - drawdown)
            
        cushion = (account_value - floor_value)/account_value
        risky_w = np.minimum(1, m*cushion)
        risky_w = np.maximum(0, risky_w)
        safe_w = 1 - risky_w

        risky_allocation = account_value*risky_w
        safe_allocation = account_value*safe_w
        account_value = risky_allocation*(1 + risky_r.iloc[step]) + safe_allocation*(1 + safe_r.iloc[step])

        # save the values in Data Frame
        cushion_hist.iloc[step] = cushion
        risky_w_hist.iloc[step] = risky_w
        account_hist.iloc[step] = account_value
        floor_hist.iloc[step] = floor_value
    
    risky_wealth = initW*(1 + risky_r).cumprod()
    cppi_results = {
        "Wealth" : account_hist,
        "Risky Wealth" : risky_wealth,
        "Risk Budget" : cushion_hist,
        "Risk Allocation" : risky_w_hist,
        "Multiplier" : m,
        "Initial Wealth" : initW,
        "Floor" : floor_hist,
        "Risky Return" : risky_r,
        "Safe Return" : safe_r
    }
    return cppi_results

def summary_stats(r, riskfree_rate=0.03, periods_per_year = 12):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(avg_return, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualized_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, rf = riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_h)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


