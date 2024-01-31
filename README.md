# Summary of all my qaunt related projects

## Introduction to Quant Finances with python
  ### 1. Review of Distributions and their moments
  #### 1.1 To declare random Distributions
    Normal -> x = np.random.standard_normal(size)
    Student t -> x = np.random.standard_t(df=coef, size=size)
    Uniform -> x = np.random.uniform(size = size)
    Exponential -> x = np.random.exponential(scale=coef, size=size)
    'Chi-squared' -> x = np.random.chisquare(df=coef, size=size)
 #### 1.2  Moments
 ##### **Definition**: certain quantitative measures related to the shape of the function's graph
    mu = np.mean(x)
    sigma = np.std(x)
    skew = skew(x) 
  skew is positive if the queue in the right is graeter than the left one,
    this is important to know the expected profitability

    kurt = kurtosis(x) 
  kurt measures the probability of black swans / home runs or the probability 
    of falling into queues
  #### 1.3. Jarque-Bera Test
  ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/d4e5e1b491ad57619afa0771f9d30b651a95bb15)

  where n is the number of observations and S & K are the skewness and kurtosis as described above 

  `jb_stat= (size/6)*(skew**2 + 1/4*(kurt)**2)`

  **P-values**
  
  A p-value is the probability of an observed result assuming that the null hypothesis (no relationship exists between the two variables being studied) is true. 

It returns the probability of having extreme events

`p_value = 1- chi2.cdf(jb_stat, df=2)`

Where chi2 is a chi-squared continuous random variable, and cdf  gives the distribution function, the integral between and the fist argument (jb_stat in our case)

Then we calculate if the random variable passed the normality test:

`is_normal = (p_value > 0.05)` That returns true if the p_value is greater then 5%

This means that we can say with 95% that it is a normal distribution.

    
  
##
##
