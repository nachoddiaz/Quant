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

### 2.4 Value at risk
#### **Definition**: VaR is a measure of the risk of loss of investment/Capital that estimates how much a set of investments might lose (with a given probability)given normal market conditions.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e2f2d2e87c32176846e5b0974f21be1a11df2666)

<p>
  Mathematically,
  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/a7a9338ad7ab25257903853f42e33a740dd47728" alt="texto_alternativo" width="80" style="vertical-align:middle; margin:0px 10px">
is the <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1dcc25f05dca60e358d4d22e8342fad5ad7affbb" alt="texto_alternativo" width="70" style="vertical-align:middle; margin:0px 10px">-quantile of  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/961d67d6b454b4df2301ac571808a3538b3a6d3f" alt="texto_alternativo" width="15" style="vertical-align:middle; margin:0px 10px"> ()
</p>

In the (1-alpha)% of the cases, my return will be greater than the VaR returned

We employ an alpha level of 5% to identify the worst trading day within a month, considering that there are approximately 252 tradable days in a calendar year. This calculation leads to an average of 21 tradable days per month. Consequently, the application of this alpha level enables us to ascertain that the worst trading day falls within the lowest 5% of all trading days in a given month.



## 3. Capital Asset Pricing Model

### 3.1 Introduction

#### **Definition**:The model takes into account the asset's sensitivity to non-diversifiable risk, often represented by the quantity beta (β) in the financial industry, as well as the expected return of the market and the expected return of a theoretical risk-free asset. Also can be defined as a linear regression of asset Alpha respect to the market M

![](https://github.com/nachoddiaz/Quant/blob/main/2024%20python%20for%20trading/img/CAPM_as_linear_regression.png) 

Where R is the total return <br>
Alpha (α) is the absolute return (return that cant be explained by the market)<br>
Beta (β) is the systematic risk (exposure of R to the market + cant be diversified away). If Beta surpasses 1, the asset is considered aggressive. If the value is exactly 1, the asset is deemed neutral. Should the value fall below 1, it is categorized as a defensive asset. <br>
Epsylon (ε) is the idiosyncratic risk (can be eliminated via diversification)<br>

#### Efficient Market Theory: The efficient-market hypothesis (EMH) is a hypothesis in financial economics that states that asset prices reflect all available information. A direct implication is that it is impossible to "beat the market" consistently on a risk-adjusted basis since market prices should only react to new information. Therefore alpha is necessarily zero

R = β*R<sub>M</sub> + ε

Taking Expectations 

E[R] = β* E[R<sub>M</sub>]     

<img src="https://github.com/nachoddiaz/Quant/blob/main/2024%20python%20for%20trading/img/Beta_%20as_VAdjCorrelation.png" alt="texto_alternativo" width="300" style="vertical-align:middle; margin:0px 10px">

Where ρ(r<sub>a</sub>, r<sub>M</sub>) is the correlation between our portfolio and the market<br>
σ<sub>a</sub> and σ<sub>M</sub> are the volatility of our portfolio and the market respectively

Therefore β is a volatility-adjusted correlation



## 3.2 Classification of investment strategies

### A. Index tracker

It replicate the performance of a benchmark (Index, commodity...)<br>
β=1, α=0

### B. Traditional long-only asset manager: 

Outperform the market with an extra, uncorrelated return<br>
β=1, α>0

### C. Smart beta:
Outperform the market by dynamically adjusting your portfolio weights<br>
β > 1 when the market is up<br>
β < 1 when the market is down<br>
α = 0<br>

### D. Hedge Fund
Deliver absolute returns that are not correlated with the market<br>
β=0, α>0

<img src="https://github.com/nachoddiaz/Quant/blob/main/2024%20python%20for%20trading/img/Inversion_models.png" alt="texto_alternativo" width="500" style="horizontal-align:middle;vertical-align:middle; margin:0px 10px"><br>


## 3.3 Linear Regression

#### **Definition**: Given a data set ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/bb65235f66e69d8c663b673c5952ee7a64e9246d)   of n statistical units, a linear regression model assumes that the relationship between the dependent variable y and the vector of regressors x is linear.<br>
This relationship is modeled through a disturbance term or error variable ε — an unobserved random variable that adds "noise" to the linear relationship between the dependent variable and regressors. Thus the model takes the form.<br>


Using the scipy.stats library, our code looks like this: <br> `beta, alpha, r, p_value, std_err = st.linregress(x, y=y, alternative='two-sided')`

Where beta and alpha are as previously defined, x represents the vector of closing prices for the benchmark asset, whereas y denotes the corresponding vector for the security asset. Meanwhile, r represents the correlation between x and y. <br>

We define $R^2$, pronounced as "R squared", as the coefficient of determination. It signifies the proportion of the variation in the dependent variable that can be predicted from the independent variable(s). Also can be defined as the part of R<sub>a</sub> that can be explained by α + β*R<sub>M</sub>

Also we calculate the null hypothesis through the p_values: 
`null_hyp = p_value > 0.05`
to know if the asset is orthogonal to the market and consequently, its covariance matrix equals zero, resulting in a beta value of zero. This eliminates systematic risk, thereby enhancing the diversification of the investment portfolio.


## 4. Hedging
### 4.1 Beta Neutral Vs Delta Neutral
  We start from defining S<sub>0</sub> as the vlaue of a given security in USD
  #### Hedge with one Security
Delta neutral: Find S<sub>1</sub> such as S<sub>0</sub> + S<sub>1</sub> = 0<br>
Beta neutral: Find S<sub>1</sub> such as β<sub>0</sub>*S<sub>0</sub> + β<sub>1</sub>*S<sub>1</sub> = 0
  #### Hedge with N Security
Delta neutral: Find S<sub>1</sub>,...,S<sub>N</sub> such as S<sub>0</sub> + ΣS<sub>n</sub> = 0<br>
Beta neutral: Find S<sub>1</sub>,...,S<sub>N</sub> such as β<sub>0</sub>S<sub>0</sub> + Σβ<sub>n</sub>S<sub>n</sub> = 0

  #### Beta and Delta Neutral Hyperplanes
  We define Delta-neutral hyperplane in R<sup>N</sup> as L<sub>delta</sub> =  {S<sub>0</sub> + ΣS<sub>n</sub> = 0} and Beta-neutral hyperplane in R<sup>N</sup> as L<sub>beta</sub> = {β<sub>0</sub>S<sub>0</sub> + Σβ<sub>n</sub>S<sub>n</sub> = 0} thus, the optimal solution would be to achieve a scenario in which our portfolio simultaneously attains Delta neutrality and Beta neutrality: P<sub>ideal</sub> = L<sub>delta</sub> ∩ L<sub>beta</sub> that is a hyperplane of dimension N-2.

  #### Example with NVDA, AAPL and MSFT
  **Theoretical explanation**<br>
  Our goal is to achieve both a delta and beta neutral portfolio starting from a 10M long position in NVDA.<br>
  β<sub>NVDA</sub> = 2.1813 <br>
  β<sub>AAPL</sub> = 1.2895 <br>
  β<sub>MSFT</sub> = 1.2621 <br>

 β<sub>NVDA</sub> > β<sub>AAPL</sub> > β<sub>MSFT</sub>

  L<sub>delta</sub> = {S<sub>NVDA</sub> + S<sub>AAPL</sub> + S<sub>MSFT</sub> = 0} <br>
  
  L<sub>beta</sub> = {β<sub>NVDA</sub>S<sub>NVDA</sub> + β<sub>AAPL</sub>S<sub>AAPL</sub> + β<sub>MSFT</sub>S<sub>MSFT</sub> = 0}

  P<sub>ideal</sub> = L<sub>delta</sub> ∩ L<sub>beta</sub>

  S<sub>AAPL</sub> < 0   &  S<sub>AAPL</sub> > 0

  **Implementation**<br>

  First we need to compute the betas of NVDA, AAPL and MSFT regarding benchamrk: <br>
  
    def betas(benchmark, security):
      m = model(benchmark, security)
      m.sync_timeseries()
      m.compute_linear_regression()
      return m.beta  
      
    def compute_betas(self):
        self.position_beta = betas(self.benchmark, self.position_ric)
        self.position_beta_usd = self.position_beta * self.position_delta_usd
        for security in self.hedge_securities:
            beta = betas(self.benchmark, security)
            self.hedge_betas.append(beta)
            
  And then we need to calclate the weights of AAPL and MSFT

     def compute_optimal_hedge(self):
        dimensions = len(self.hedge_securities)
        if dimensions != 2:
            print('Cannot compute the exact solution cause dimensions = ' + str(dimensions))
            return
        deltas = np.ones([dimensions])
        target = -np.array([self.position_delta_usd, self.position_beta_usd]) 
        #First we put the 2 arrays as columns and then into rows
        mtx = np.transpose(np.column_stack((deltas, self.hedge_betas)))          
        self.hedge_weights = np.linalg.solve(mtx,target)

  Mathematically the matrix would be such that:

        [ 1  1  ]  * [ S1 ]        [  -S0   ]
        [ β1 β2 ]    [ S2 ]    =   [ -β0*S0 ]
  

Utilizing this method enables us to hedge our principal security exclusively with two additional assets.     


### 4.2 Generalizing the coverage model

The present model necessitates a substantial financial outlay to address a relatively minor sum (i.e., $660 M significantly exceeds $10 M). Consequently, we are prepared to compromise on our beta value (value inside the other two) to obtain coverage at a reduced cost.<br> 

Where N>2 there are infinitely many solutions to the problem f(x) = 0 so we are interested in the solution with the **smallest weights** ->  with the smallest norm ||x||

We define x, β and I as follows
         
        [ S1  ]        [ β1  ]          [  1  ]
    x = [ ... ]    β = [ ... ]      I = [ ... ]
        [ SN  ]        [ βN  ]          [  1  ]
  

Also f(x; ε) = f<sub>delta</sub>(x) + f<sub>beta</sub>(x) + f<sub>penalty</sub>(x; ε)<br>

Where:<br>
    f<sub>delta</sub>(x) = (I<sup>T</sup>x + S<sub>0</sub>)<sup>2</sup>,<br>
    f<sub>beta</sub>(x) = (β<sup>T</sup>x + β<sub>0</sub>S<sub>0</sub>)<sup>2</sup> <br> f<sub>penalty</sub>(x; ε) = ε||x||<sup>2</sup><br>
    
The parameter ε will allow the portfolio manager to control the degree of “onesidedness” (the securities we use to hedge must have the sime sign) of the optimal solution:<br>
. No constraints -> we solve the problem without the need of Lagrange Multipliers<br>
. The solution cannot be perfectly beta-neutral and delta-neutral, as before<br>


The code that executes the coverage model generalizing without error (ε) is as follows: <br>

    def cost_function(x, betas, target_delta, target_beta):
        dimension = len(x)
        deltas = np.ones([dimension])
        f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2
        f_beta = (np.transpose(betas).dot(x).item() + target_beta)**2
        f = f_delta + f_beta
        return f
        
    #initial condition
    x0 = -target_delta/len(betas) * np.ones(len(betas))
    optimal_result = op.minimize(fun=cost_function, x0=x0, args=(betas,target_delta,target_beta))

Where `betas` is β, `target_delta` is S<sub>0</sub> and `target_beta` is β<sub>0</sub>S<sub>0</sub><br>

If we add the error (ε) the code the code would look like this: <br>

f<sub>penalty</sub>(x; ε) = ε||x||<sup>2</sup> = ` f_penalty = regularisation * np.sum(x**2) `<br>

where `np.sum(x**2)` is ||x||^2<br>

And now the funciont will look like this `f = f_delta + f_beta + f_penalty`

Now we want to order all the assets of a set based on their correlation with the asset to be analyzed to choose the "n" that best suit us 

We can do it by creating a dataFrame and order the items by it correlation<br>
Code ->

        def dataframe_correl_beta (benchmark, position_security, hedge_universe):
        decimals = 5
        df = pd.DataFrame()
        correlations = []
        betas = []
        for hedge_security in hedge_universe:
            correlation = compute_correlation(position_security, hedge_security)
            beta = compute_betas(benchmark, hedge_security)
            correlations.append(np.round(correlation, decimals))
            betas.append(np.round(beta, decimals))
        df['hedge_security'] = hedge_universe
        df['correlation'] = correlations
        df['beta'] = betas
        df = df.sort_values(by='correlation', ascending=False)
        return df

Declaring the universe of assets and compare then by: <br>

      df = capm.dataframe_correl_beta(benchmark, position_security, hedge_universe_fin)



## 5. Factor Investing

### 5.1 Introduction

#### **Definition**: Factor investing is an investment approach that involves targeting quantifiable firm characteristics or “factors” that can explain differences in stock returns. Security characteristics that may be included in a factor-based approach include size, low-volatility, value, momentum, asset growth, profitability, leverage, term and carry

#### Code
We aim to compute the correlation and beta of a security in relation to each factor. This is because, according to a rule of thumb, correlations reveal which factors affect the asset, and betas determine the extent to which each factor influences the asset.

    def dataframe_factors (security, factors):
    decimals = 5
    df = pd.DataFrame()
    correlations = []
    betas = []
    for factor in factors:
        correlation = compute_correlation(security, factor)
        beta = compute_betas(factor,security)
        correlations.append(np.round(correlation, decimals))
        betas.append(np.round(beta, decimals))
    df['factors'] = factors
    df['correlation'] = correlations
    df['beta'] = betas
    df = df.sort_values(by='correlation', ascending=False)
    return df 


Where `compute_correlation` is:

    def compute_correlation (security1 , security2):
    m = model(security1, security2)
    m.sync_timeseries()
    m.compute_linear_regression() 
    return m.correlation

And where `compute_beta` is:

    def compute_betas (benchmark, security):
    m = model(benchmark, security)
    m.sync_timeseries()
    m.compute_linear_regression()
    return m.beta

It is important to note that within the compute_betas function, the term "security" assumes the role of a security, and "factor" takes the position of the benchmark.





## 6. Geometry of the variance-covariance matrix

### 6.1. Mathematical Introduction

Starting from the general equation of an ellipse : Ax<sup>2</sup> + 2Bxy + Cy<sup>2</sup> + Dx  Ey + F = 0<br> 
if we make a rotation, a translation and we refactore it, the ecuation looks (<sup>x</sup>/<sub>a</sub>)<sup>2</sup> + (<sup>y</sup>/<sub>b</sub>)<sup>2</sup> = 1 <br<
In matrix notation it becomes [x, y] * |λ₁  0|&nbsp;&nbsp;&nbsp;&nbsp;|x| <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|0  λ₂| * |y| = 1 <br>
  where λ₁ = 1/a² y λ₂ = 1/b²

The theorem that we apply is that a change of variable can be applied to any symmetric (a<sub>ij</sub> = a<sub>ji</sub>) or positive semi-definite (x<sup>T</sup> Qx ≥ 0 for any x ε R<sup>N</sup>)  matrix and it can be diagonalized.

  Under the new coordinates  |A  B|  ->   |λ₁  0|<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|B  C|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|0  λ₂|

  ### 6.2. Eigenvalues & Eigenvectors

With that notation, we define Q as | λ1   0   ...   0 |<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| 0   λ2   ...   0 |<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| ...  ... ... ...  |<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| 0    0   ...  λn |
<br>
Where λ<sub>n</sub> are the eigenvalues and the vector associated to them is the eigenvector Qv<sub>n</sub> = λ<sub>n</sub>v<sub>n</sub><br>


### 6.3. Variance-covariance matrix

Let 𝑋<sub>1</sub>, … , 𝑋<sub>n</sub> be random variables with means 𝜇<sub>1</sub>, … , 𝜇<sub>n</sub> resp<br>
The variance-covarianze matrix Q is such that Q<sub>ij</sub> = Cov(X<sub>i</sub>, X<sub>j</sub>) = E[(X<sub>i</sub> -  𝜇<sub>i</sub>)(X<sub>j</sub> -  𝜇<sub>j</sub>)]<br>
Q by definition, is symmetric -> Q<sub>ij</sub> = Q<sub>ji</sub> and always positive semi-definite<br>
Defining: <br>

        [ X1  ]        [ 𝜇1  ]
    x = | ... |    𝜇 = | ... |
        [ XN  ]        [ 𝜇N  ]
Then Q(X) = E[(X<sub>i</sub> -  𝜇<sub>i</sub>)(X<sub>j</sub> -  𝜇<sub>j</sub>)<sup>T</sup>]<br>
And for any w ε R<sup>N</sup> we have w<sup>T</sup>Q(X)w = E[w<sup>T</sup>(X -  𝜇)(X -  𝜇)<sup>T</sup>w] = E[((X -  𝜇)<sup>T</sup>w)<sup>2</sup>] ≥ 0


### 6.4. Implementation in code
 We are going to follow the next three steps: <br>
 
 Get the intersection of all timestamps:
 
    df = pd.DataFrame()
    #Like a mapping in solidity
    dic_timeseries = {}
    timestamps=[]
    for ric in rics:
        t = market_data.load_timeseries(ric)
        dic_timeseries[ric] = t
        if len(timestamps) == 0:
            timestamps = list(t['date'].values)
        temp_timestamps = list(t['date'].values)
        timestamps = list(set(timestamps) & set(temp_timestamps))


  Sync timeseries

    for ric in dic_timeseries:
        t = dic_timeseries[ric]
        t = t[t['date'].isin(timestamps)]
        t = t.sort_values(by='date', ascending=True)
        t = t.dropna()
        t = t.reset_index(drop=True)
        dic_timeseries[ric] = t
        if df.shape[1] == 0:
            df['date'] = timestamps
        df[ric] = t['return']
          

  Compute Variance-covariance matrix and Correlation matrix:

    mtx= df.drop(columns=['date'])
    mtx_var_cov = np.cov(mtx, rowvar=False)
    mtx_correl = np.corrcoef(mtx, rowvar=False)











  




