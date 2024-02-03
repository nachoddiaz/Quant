# Introduction to Quant Finances with python
  ## 1. Review of Distributions and their moments
  ### 1.1 To declare random Distributions
    Normal -> x = np.random.standard_normal(size)
    Student t -> x = np.random.standard_t(df=coef, size=size)
    Uniform -> x = np.random.uniform(size = size)
    Exponential -> x = np.random.exponential(scale=coef, size=size)
    'Chi-squared' -> x = np.random.chisquare(df=coef, size=size)
 ### 1.2  Moments
 #### **Definition**: certain quantitative measures related to the shape of the function's graph
    mu = np.mean(x)
    sigma = np.std(x)
    skew = skew(x) 
  skew is positive if the queue in the right is graeter than the left one,
    this is important to know the expected profitability

    kurt = kurtosis(x) 
  kurt measures the probability of black swans / home runs or the probability 
    of falling into queues
  ### 1.3. Jarque-Bera Test
  ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/d4e5e1b491ad57619afa0771f9d30b651a95bb15)

  where n is the number of observations and S & K are the skewness and kurtosis as described above 

  `jb_stat= (size/6)*(skew**2 + 1/4*(kurt)**2)`

  ### 1.4 P-Values
  
  A p-value is the probability of an observed result assuming that the null hypothesis (no relationship exists between the two variables being studied) is true. 

It returns the probability of having extreme events

`p_value = 1- chi2.cdf(jb_stat, df=2)`

Where chi2 is a chi-squared continuous random variable, and cdf  gives the distribution function, the integral between and the fist argument (jb_stat in our case)

Then we calculate if the random variable passed the normality test:

`is_normal = (p_value > 0.05)` That returns true if the p_value is greater then 5%

This means that we can say with 95% that it is a normal distribution.

    
## 2. Real Data Analysis

### 2.1 Sharpe Ratio

#### **Definition**: Sharpe ratio measures the performance of an investment such as a security or portfolio compared to a risk-free asset. For each unit of risk assumed, X extra return will be earned

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/d54973db1901fd6f25c55d4bb88fddc75b0fe09f)

We look for a Sharpe ratio of at least 2 since the confidence interval returns positive.


### 2.2 Time Series Representation

    ric = 'your_ticker'
    directory = 'your_path'
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True, format='%Y-%m-%d')
    t['close'] =raw_data['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return close'] = t['close']/t['close_previous'] -1
    t = t.dropna()
    t= t.reset_index(drop=True)
    plt.figure()
    t.plot(kind='line', x='date', y='close', grid=True, color='blue',label=ric, title='Timeseries of close prices for '+ ric)
    plt.show()


### 2.3 Normality test on all assets

    import os
    for file_name in os.listdir(directory):
    print('file_name = ' + file_name)
    #returns the first element after split
    ric = file_name.split('.')[0]
    #get data_frame
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True, format='mixed')
    t['close'] =raw_data['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return close'] = t['close']/t['close_previous'] -1
    t = t.dropna()
    t= t.reset_index(drop=True)
    sim = random_variables.simulator(inputs)
    sim.x = t['return close'].values
    sim.inputs.size = len(sim.x)
    sim.str_title = sim.inputs.random_var_type
    sim.compute_stats()
    #generate lists
    rics.append(ric)
    is_normals.append(sim.is_normal)
    
    df = pd.DataFrame()
    df['ric'] = rics
    df['is_normal'] = is_normals

##
##