
# coding: utf-8

# In[3]:


from numpy import matrix, array, zeros, empty, sqrt, ones, dot, append, mean, cov, transpose, linspace
from numpy.linalg import inv, pinv
import numpy as np
import scipy.optimize
import math

# This algorithm performs a Black-Litterman portfolio construction. The framework
# is built on the classical mean-variance approach, but allows the investor to 
# specify views about the over- or under- performance of various assets.
# Uses ideas from the Global Minimum Variance Portfolio 
# algorithm posted on Quantopian. Ideas also adopted from 
# http://www.quantandfinancial.com/2013/08/portfolio-optimization-ii-black.html.

# Compute the expected return of the portfolio.
def compute_mean(W,R):
    return sum(R*W)

# Compute the variance of the portfolio.
def compute_var(W,C):
    return dot(dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation. 
def compute_mean_var(W, R, C):
    return compute_mean(W, R), compute_var(W, C)

# objective function you can add other constrains or targets here
def fitness(W, R, C, r, daily_return, asset_beta):
    # For given level of return r, find weights which minimizes portfolio variance.
    mean_1, var = compute_mean_var(W, R, C)
    # Penalty for not meeting stated portfolio return effectively serves as optimization constraint
    # Here, r is the 'target' return
    cum_p = compute_historical_performance(daily_return, W)
    drawdown = max_drawdown(cum_p)
    
    penalty = 100*abs(mean_1-r)
    penalty1 = 1*np.maximum(drawdown-0.12,0)
    penalty2 = 1*np.maximum(W@asset_beta-1.5,0)
    return var + penalty + penalty1+penalty2

# Solve for optimal portfolio weights
def solve_weights(R, C, rf, r_target, daily_return, asset_beta):
    n = len(R)
    W = ones([n])/n # Start optimization with equal weights
    b_ = [(0.00,0.4) for i in range(n)] # Bounds for decision variables
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. }) # Constraints - weights must sum to 1
    # 'target' return is the expected return on the market portfolio
    optimized = scipy.optimize.minimize(fitness, W, (R, C, r_target, daily_return, asset_beta), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x     
        
# Weights - array of asset weights (derived from market capitalizations)
# Expreturns - expected returns based on historical data
# Covars - covariance matrix of asset returns based on historical data
def assets_meanvar(daily_returns):    
    
    # Calculate expected returns
    expreturns = array([])
    (rows, cols) = daily_returns.shape
    for r in range(rows):
        expreturns = append(expreturns, mean(daily_returns[r]))
    
    # Compute covariance matrix
    covars = cov(daily_returns)
    # Annualize expected returns and covariances
    # Assumes 255 trading days per year    
    expreturns = (1+expreturns)**255-1
    covars = covars * 255
    
    return expreturns, covars

def compute_historical_performance(daily_returns ,W):
    ret_p = np.log(1+W@daily_returns)
    cum_p = np.exp(np.cumsum(ret_p))
    return cum_p

def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd


# In[4]:


import numpy as np
import pandas as pd
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
params = [
    {
    'q': "VFV", 
    'x': "TSE", 
    },
    {
    'q': "IEUR", 
    'x': "NYSEARCA", 
    },
    {
    'q': "JPNH", 
    'x': "EPA", 
    },
    {
    'q': "ASHR", 
    'x': "NYSEARCA", 
    },
    {
    'q': "LYAEJ-USD", 
    'x': "SWX", 
    },
    {
    'q': "EEM", 
    'x': "NYSEARCA", 
    },
    {
    'q': "SSFCX", 
    'x': "MUTF", 
    },
    {
    'q': "JNK", 
    'x': "NYSEARCA", 
    },
    {
    'q': "ZROZ", 
    'x': "NYSEARCA", 
    },
    {
    'q': "EMB", 
    'x': "BMV", 
    }
    ]
period = '10Y'
# get price data (return pandas dataframe)
df = get_prices_data(params, period)
print(df)


# In[183]:


asset_list = ['VFV_Close','IEUR_Close','JPNH_Close','ASHR_Close','LYAEJ-USD_Close', 'EEM_Close', 'SSFCX_Close', 'JNK_Close','ZROZ_Close','EMB_Close']
daily_price = df.fillna(method='ffill')[asset_list]


# In[184]:


daily_return = daily_price.pct_change().dropna()
print(daily_return)
etf_return = daily_return


# In[186]:


import numpy as np
import pandas as pd
df = pd.read_excel('xlsx/benchmark.xlsx', sheetname = 'Daily_Index_Price')
# use data from 2000.1.1 onward
df_sub = df.iloc[2:]
df_new = pd.DataFrame(index = pd.to_datetime(df_sub.index))
for i in range(df_sub.shape[1]):
    df_new['asset'+str(i+1)]= pd.to_numeric(df_sub.iloc[:,i].values)
    # partially clear data, ideally trading days, but here weekdays
df_weekday = df_new[(df_new.index.weekday<5)]
# use data from 2015-10-1 onwards
return_index = df_weekday[df_weekday.index>'2009-12-31']
return_index = return_index.values
# compute daily return
daily_return = (return_index[1:,:]-return_index[:-1,:])/return_index[:-1,:]
index_return = pd.DataFrame(index = df_weekday[df_weekday.index>'2009-12-31'].index[1:], data = daily_return)
print(index_return)


# In[187]:


daily_return = index_return.values.transpose()
expreturns, covars = assets_meanvar(daily_return)
R = expreturns # R is the vector of expected returns
C = covars # C is the covariance matrix
print(R)
print(np.diag(C))


# In[188]:


from scipy.stats import linregress
asset_beta = np.zeros(daily_return.shape[0])
for i in range(daily_return.shape[0]): 
    asset_beta[i], alpha, r, _, _ =  linregress(daily_return[0,:],daily_return[i,:])
print(asset_beta)


# In[189]:


rf = 0.01
# use historical data as prior mean and cov
Pi = R - rf

# use capm

benchmark_weight = np.array([0.15,0.05,0.05,0.05,0.15,0.1,0.15,0.1,0.15,0.05])

new_mean = compute_mean(benchmark_weight,R)
new_var = compute_var(benchmark_weight,C)


lmb = (new_mean - rf) / new_var # Compute implied equity risk premium
Pi = dot(dot(lmb, C), benchmark_weight) # Compute equilibrium excess returns
print(Pi+rf)

# Solve for weights before incorporating views (example target = 0.05)
W = solve_weights(Pi+rf, C, rf, r_target = new_mean, daily_return = daily_return, asset_beta = asset_beta)
mean_prior, var_prior = compute_mean_var(W, Pi+rf, C) 

cum_p = compute_historical_performance(daily_return, W)
drawdown = max_drawdown(cum_p)

print('the prior new weights is')
print(W)
print('the prior portfolio expected return is')
print(mean_prior)
print('the prior portfolio expected volatility is')
print(np.sqrt(var_prior))
print('the historical max_drawdown is')
print(drawdown)
print('the portfolio beta is')
print(W@asset_beta)


# In[201]:


# use historical data as prior mean and cov
Pi = R - rf
print('historical average return')

benchmark_weight = np.array([0.15,0.05,0.05,0.05,0.15,0.1,0.15,0.1,0.15,0.05])

new_mean = compute_mean(benchmark_weight,R)
new_var = compute_var(benchmark_weight,C)


lmb = (new_mean - rf) / new_var # Compute implied equity risk premium
Pi = dot(dot(lmb, C), benchmark_weight) # Compute equilibrium excess returns
print(Pi+rf)

target_return = 0.06

# Solve for weights before incorporating views (example target = 0.05)
W = solve_weights(Pi+rf, C, rf, r_target = target_return, daily_return = daily_return, asset_beta = asset_beta)
mean_prior, var_prior = compute_mean_var(W, Pi+rf, C) 

cum_p = compute_historical_performance(daily_return, W)
drawdown = max_drawdown(cum_p)

print('the prior new weights is')
print(W)
print('the prior portfolio expected return is')
print(mean_prior)
print('the prior portfolio expected volatility is')
print(np.sqrt(var_prior))
print('the historical max_drawdown is')
print(drawdown)
print('the portfolio beta is')
print(W@asset_beta)


# In[191]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))

font = {'family' : 'Serif',
         'weight': 'normal',
         'size'  : 11}
mpl.rc('font', **font)
# a4 size: 8.27,11.69
fig = plt.figure(figsize=(8.27,3.5))
ax = fig.add_subplot(111)

ax.plot(range(cum_p.shape[0]), cum_p, c = 'xkcd:ocean blue')
ax.set_xlabel('days')
ax.set_ylabel('cumulative return')
ax.set_title('portfolio historical performance (initial value 1), index')

plt.legend(loc='best') 
plt.show()     


# In[196]:


cum_p = compute_historical_performance(etf_return.values.transpose(), W)
drawdown = max_drawdown(cum_p)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))

font = {'family' : 'Serif',
         'weight': 'normal',
         'size'  : 11}
mpl.rc('font', **font)
# a4 size: 8.27,11.69
fig = plt.figure(figsize=(8.27,3.5))
ax = fig.add_subplot(111)

ax.plot(range(cum_p.shape[0]), cum_p, c = 'xkcd:ocean blue')
ax.set_xlabel('days')
ax.set_ylabel('cumulative return')
ax.set_title('portfolio historical performance (initial value 1), etf')

plt.legend(loc='best') 
plt.show()     


# In[194]:


# plot the prior m-v frontier
n = 80
plot_mean = np.zeros(n)
plot_var = np.zeros(n)
for i in range(n):
    r_target = 0.02+i*0.001
    W = solve_weights(Pi+rf, C, rf, r_target = r_target, daily_return = daily_return, asset_beta = asset_beta)
    plot_mean[i], plot_var[i] = compute_mean_var(W, Pi+rf, C) 


# In[195]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))

font = {'family' : 'Serif',
         'weight': 'normal',
         'size'  : 11}
mpl.rc('font', **font)
# a4 size: 8.27,11.69
fig = plt.figure(figsize=(8.27,3.5))
ax = fig.add_subplot(111)

ax.plot(plot_var, plot_mean, c = 'xkcd:ocean blue')
ax.set_xlabel('variance')
ax.set_ylabel('mean')
ax.set_title('prior mean-variance frontier')

plt.legend(loc='best') 
plt.show()            


# In[ ]:


# VIEWS ON ASSET PERFORMANCE
# asset 1 will out perform asset 2 by 3%, and
# that asset 1 will outperform asset 3 by 2%.
# that asset 5 will under outperform the benchmark by 5%
P = np.array([[1,-1,0,0,0], [1,0,-1,0,0], [0,0,0,0,1]])  
Q = np.array([0.03,0.02, 0.05])


## note adjust tau to adjust your confidence about your view
tau = 0.025 # tau is a scalar indicating the uncertainty 
# in the CAPM (Capital Asset Pricing Model) prior
omega = dot(dot(dot(tau, P), C), transpose(P)) # omega represents 
# the uncertainty of our views. Rather than specify the 'confidence'
# in one's view explicitly, we extrapolate an implied uncertainty
# from market parameters.

# Compute equilibrium excess returns taking into account views on assets
sub_a = inv(dot(tau, C))
sub_b = dot(dot(transpose(P), inv(omega)), P)
sub_c = dot(inv(dot(tau, C)), Pi)
sub_d = dot(dot(transpose(P), inv(omega)), Q)
Pi_new = dot(inv(sub_a + sub_b), (sub_c + sub_d))         
# Perform a mean-variance optimization taking into account views          

new_weights = solve_weights(Pi_new + rf, C, rf, r_target = 0.08, daily_return = daily_return)
mean_new, var_new = compute_mean_var(new_weights, Pi_new + rf, C)
print('the posterior new weights is')
print(new_weights)
print('the posterior portfolio expected return is')
print(mean_new)
print('the posterior portfolio expected volatility is')
print(np.sqrt(var_new))


# In[ ]:


# plot the posterior m-v frontier
n = 80
plot_mean = np.zeros(n)
plot_var = np.zeros(n)
for i in range(n):
    r_target = 0.02+i*0.001
    new_weights = solve_weights(Pi_new+rf, C, rf, r_target = r_target, daily_return = daily_return)
    plot_mean[i], plot_var[i] = compute_mean_var(new_weights, Pi_new + rf, C) 


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))

font = {'family' : 'Serif',
         'weight': 'normal',
         'size'  : 11}
mpl.rc('font', **font)
# a4 size: 8.27,11.69
fig = plt.figure(figsize=(8.27,3.5))
ax = fig.add_subplot(111)

ax.plot(plot_var, plot_mean, c = 'xkcd:ocean blue')
ax.set_xlabel('variance')
ax.set_ylabel('mean')
ax.set_title('posterior mean-variance frontier')

plt.legend(loc='best') 
plt.show()            

