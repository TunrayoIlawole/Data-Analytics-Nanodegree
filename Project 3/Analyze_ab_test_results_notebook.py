#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# 
# <a id='probability'></a>
# #### Part I - Probability
# 

# In[1]:


# First we import our libraries.
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(42)


# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


# Then we read the data
df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


# Find the nummber of rows and columns in the dataset.
df.shape


# c. The number of unique users in the dataset.

# In[4]:


# Find the number of unique values in the dataset.
df.nunique()


# d. The proportion of users converted.

# In[5]:


# Find the proportion of users that converted from using the old landing page to the new landing page.
(df.converted == 1).mean()


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


df.query("landing_page == 'new_page' and group != 'treatment'").count()[0] + df.query("landing_page != 'new_page' and group == 'treatment'").count()[0]


# f. Do any of the rows have missing values?

# In[7]:


# To find out if there are missing values the rows of the dataset
df.isnull().sum()


# In[8]:


# Create a new and more acceptable dataset where "new_page" and "treatment" match.
df_new = df.query("landing_page == 'new_page' and group == 'treatment'")
df_old = df.query("landing_page == 'old_page' and group == 'control'")

df2 = pd.concat([df_new, df_old])
df2.tail()


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# a. How many unique **user_id**s are in **df2**?

# In[10]:


# Check the number of unique users in the new dataset.
df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


# Information about the one non-unique user.
duplicateRow = df2[df2.duplicated(['user_id'])]
print(duplicateRow)


# c. What is the row information for the repeat **user_id**? 

# In[12]:


print(df.loc[[2893]].info())


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


# Remove the non-unique user row.
df2 = df2.drop(2893, axis = 0)


# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


# Probability of an individual converting regardless of the page they receive
df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


# Probability that an individual in the control group converted
df2[df2['group'] == 'control']['converted'].mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


# Probability that an individual in the treatment group converted
df2[df2['group'] == 'treatment']['converted'].mean()


# d. What is the probability that an individual received the new page?

# In[17]:


# Probability that an individual recieved the new landing page.
len(df2[df2['landing_page'] == 'new_page']) / len(df2)


# # There is no sufficient evidence to conclude that the new treatment page leads to more conversions because the proportion of people in the control group that converted is 12.03% and the proportion of people in the treatment group that converted is 11.88%. There is not much difference between the two proportions, so we cannot say for sure that one leads to more conversions. We need to undergo further testing.

# <a id='ab_test'></a>
# ### Part II - A/B Test  

# # null = 𝑝𝑜𝑙𝑑 - 𝑝𝑛𝑒𝑤 ≥ 0
# # alternate = 𝑝𝑜𝑙𝑑 - 𝑝𝑛𝑒𝑤 < 0

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[18]:


# Conversion rate for Pnew under the null
p_new = df2['converted'].mean()
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[19]:


# Conversion rate for Pold under the null
p_old = df2['converted'].mean()
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[20]:


# The number of individuals in the treatment group
n_new = df2.query("group == 'treatment'").count()[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[21]:


# The number of individuals in the control group
n_old = df2.query("group == 'control'").count()[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[22]:


new_page_converted = np.random.binomial(n_new,p_new,10000)/n_new


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[23]:


old_page_converted = np.random.binomial(n_old,p_old,10000)/n_old


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[24]:


# Pnew - Pold
new_page_converted.mean() - old_page_converted.mean()


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[25]:


p_diffs = []
new_page_converted = np.random.binomial(n_new,p_new,10000)/n_new
old_page_converted = np.random.binomial(n_old,p_old,10000)/n_old
p_diffs = new_page_converted - old_page_converted
p_diffs


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[26]:


plt.hist(p_diffs);


# # The plot is a normal distribution, as i expected.

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[27]:


# Difference between the proportion of people in the treatment group that converted and the proportion of people in the control group that converted
actual_diff = df2.query("group == 'treatment'")['converted'].mean() - df2.query("group == 'control'")['converted'].mean()
actual_diff


# In[28]:


plt.hist(p_diffs);
plt.axvline(actual_diff, color = 'red');


# In[29]:


p_diffs = np.array(p_diffs)
p_val = (p_diffs > actual_diff).mean()
p_val


# # I just computed the proportion of p_diffs greater than the actual difference. In scientific studies, this is called the "p-value". I obtained a value of 0.9028, which is greater than 0.05. Therefore, we cannot reject the null hypothesis. This means that the new page is not better than the old page. There is no difference between the two pages.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[30]:


import statsmodels.api as sm

# Number of conversions for each page
convert_old = df2.query("landing_page == 'old_page' and converted == 1").count()[0]
convert_new = df2.query("landing_page == 'new_page' and converted == 1").count()[0]

# Number of individuals who recieved each page
n_old = df2.query("landing_page == 'old_page'").count()[0]
n_new = df2.query("landing_page == 'new_page'").count()[0]
p1_hat = convert_new / n_new
p2_hat = convert_old / n_old
p = (convert_old + convert_new) / (n_old + n_new)
z_stat = (p2_hat - p1_hat) / np.sqrt(p*(1-p) * ((1 / n_old) + (1 / n_new)))
z_stat


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[31]:


from statsmodels.stats.proportion import proportions_ztest
count = np.array([n_new, n_old])
nobs = np.array([convert_new, convert_old])
value = 0.05
stat, pval = proportions_ztest(count, nobs, value = value)


# In[32]:


# Critical value
from scipy.stats import norm
norm.ppf(1 - value)


# # The p-value is 0.903 and z-score is approximately 1.31. As calculated above, the critical value is 1.64, which is greater than the z-score. Therefore, we cannot reject the null hypothesis. There is no difference between the old page and the new page. The conclusion is the same as above.

# <a id='regression'></a>
# ### Part III - A regression approach 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# # Logistic regression.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[33]:


# Create a dummy variable column for which page each user received.
df2['intercept'] = 1
df2[['drop', 'ab_page']] = pd.get_dummies(df2['group'])
df2.drop('drop', axis = 1, inplace = True)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[34]:


from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
result = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']]).fit()


# In[35]:


result.summary()


# The p-value associated with the ab_page is 0.190. It differs from the value found in part II because they have different hypothesis.
# 
# Part II Hypothesis:
# 
#                     Hnull : Pold ≥ Pnew
#                     
#                     Halternate: Pold < Pnew
# Part III Hypothesis:
# 
#                     Hnull: Pold = Pnew
#                     
#                     Halternate: Pold  ≠ Pnew

# # It is a good idea to consider other factors to add to my regression model because many different factors may affect the result. It they are not considered, the result may not be correct. 
# 
# # There are some disadventages of adding additional terms into my regression model. One of them is multicollinearility, where one factor is related to another.

# # Now we're going to consider the factor of the country where the user lives.

# In[36]:


# Read the countries dataset
country_df = pd.read_csv('countries.csv')
country_df.head()


# In[37]:


# Merge the countries dataset with the df2 dataset
complete_df = df2.merge(country_df, on = 'user_id', how = 'inner')
complete_df.head()


# In[38]:


complete_df = complete_df.join(pd.get_dummies(complete_df['country']))
complete_df.head()


# In[39]:


result = sm.Logit(complete_df['converted'], complete_df[['intercept', 'ab_page', 'CA', 'UK']]).fit()
result.summary()


# The p-values above are all greater than 0.05, so we do not reject the null hypothesis. Even if we take country as a factor into account, the result remains the same.

# # We would now like to look at an interaction between page and country to see if there significant effects on conversion.

# In[40]:


# Create a column that combines the group column and the country column
complete_df['group_country'] = complete_df['group'] + '_' + complete_df['country']
complete_df.head()


# In[41]:


complete_df = complete_df.join(pd.get_dummies(complete_df['group_country']))
complete_df.head()


# In[42]:


result = sm.Logit(complete_df['converted'], complete_df[['intercept', 'ab_page', 'CA', 'UK', 'treatment_CA', 'treatment_UK']]).fit()
result.summary()


# # All the p-values above are greater than the critical value, so there is no significant effect on the result.

# # Conclusion
# 
# In this experiment, it was our aim to find out if the landing page significantly affected the conversion rate. Hence we had our null hypothesis which stated that the old landing page has the same or even higher converted rate than the new landing page. And we had our alternate hypothesis which stated that the new landing page had a higher converted rate than the old landing page.
# 
# In order to get the result, we performed an A/B test by using two methods:
# 
# 1) Simulating from the null
# 
# 2) Calculating the z-score.
# 
# Both methods gave the same result and conclusion of not rejecting the null hypothesis: There is no significant effect of the landing page on the converted rate.
# 
# We also tried the regression method by using a logistic regression model. The p-value was different from what we obtained in the A/B test because of the difference in the null and alternate hypothesis. However, it gave a similar result to that from the A/B test.
# 
# Finally, in order to avoid the situation in Simpson's paradox, we introduced an additonal factor into the regression model. The factor introduced was the country a user lives. We looked at the individual factors and the interation of country and landing_page to see if they have significant effects on conversion. The results showed that the factors of landing page and country have no significant effect on the converted rate individually, as well as interactively.

# In[43]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




