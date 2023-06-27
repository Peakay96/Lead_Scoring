#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Leads.csv")
df. head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.columns


# ### Exploring the Data
# <hr/>
# 
# The dataframe has 9240 records with 37 features. We will first look into the percentages of errors or NANs in each of the features, then try to correct these imperfections either by dropping or imputing values. Let's get into this.

# In[7]:


feature_list = df.columns.to_list() # creating a list of features that might come in handy later on in the process


# In[8]:


df.head(3)


# In[9]:


(df[feature_list].isnull().sum()/len(df))*100 # calculating the percentages of nan values per column


# In[10]:


# making a list of features that have 25 percent or more empty values
features_greater_than_25 = []
for feature in feature_list:    
    if ((df[feature].isnull().sum()/len(df))*100) >=25:
        features_greater_than_25.append(feature)


# In[11]:


features_greater_than_25


# ### Exploring all features with more than 25 percent null values to better understand them

# In[12]:


df25 = df[features_greater_than_25]


# In[13]:


df25.head()


# In[14]:


for feature in features_greater_than_25:
    print(df25[feature].value_counts())
    print("Null Percentage : ", (df25[feature].isnull().sum()/len(df))*100)
    print("Null Count : ", df25[feature].isnull().sum())
    print('-'*50)


# #### For this case study, I am going to drop features with more than 40 percent null values.

# In[15]:


# creating a list of features with 40 percent or more null values
features_greater_than_40 = []
for feature in feature_list:    
    if ((df[feature].isnull().sum()/len(df))*100) >=40:
        features_greater_than_40.append(feature)


# In[16]:


features_greater_than_40


# In[17]:


# dropping these features
df = df.drop(features_greater_than_40, axis=1)


# In[18]:


df.columns


# In[19]:


feature_list = df.columns.to_list()
null_features = []
for feature in feature_list:    
    if ((df[feature].isnull().sum()/len(df))*100) >0:
        null_features.append(feature)
null_features


# In[20]:


# exploring each feature individually to handle null values
df['Lead Source'].value_counts()


# In[21]:


df['Lead Source'] = df['Lead Source'].fillna('Unknown')


# In[22]:


df['Lead Source'].isnull().sum()


# In[23]:


df['Lead Source'].value_counts()


# In[24]:


df['TotalVisits'].describe()


# In[25]:


df['TotalVisits'].value_counts()


# In[26]:


df.TotalVisits.median()


# In[27]:


# imputing the null values with the median since mean is a float number which won't make sense for the number of visits
df['TotalVisits'] = df['TotalVisits'].fillna(df.TotalVisits.median())


# In[28]:


df['TotalVisits'].describe() # imputing the median has not changed the mean and the std by much


# In[29]:


df['Page Views Per Visit'].describe()


# In[30]:


# I am going to use mean to impute the missing values in Page_views_per_visit
df['Page Views Per Visit'] = df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].mean())


# In[31]:


df['Page Views Per Visit'].describe()


# In[32]:


null_features


# In[33]:


print(df['Last Activity'].value_counts())
print(df['Last Activity'].isnull().sum())


# In[34]:


df['Last Activity'] = df['Last Activity'].fillna('Unknown')


# In[35]:


print(df['Last Activity'].value_counts())
print(df['Last Activity'].isnull().sum())


# In[36]:


print(df['Country'].value_counts())
print(df['Country'].isnull().sum())


# In[37]:


# clubbing Countries under two categories, India and Other
df['Country'] = df['Country'].apply(lambda x : "India" if x == 'India' else 'Other')


# In[38]:


df['Country'].value_counts()


# In[39]:


df['Country'].value_counts()


# In[40]:


print(df['Specialization'].value_counts())
print(df['Specialization'].isnull().sum())


# In[41]:


# converting select to Unknown and then adding null values to this category
df["Specialization"].replace('Select', 'Unknown', inplace = True)


# In[42]:


print(df['Specialization'].value_counts())
print(df['Specialization'].isnull().sum())


# In[43]:


df.Specialization = df.Specialization.fillna('Unknown')


# In[44]:


print(df['Specialization'].value_counts())
print(df['Specialization'].isnull().sum())


# In[45]:


print(df['How did you hear about X Education'].value_counts())
print(df['How did you hear about X Education'].isnull().sum())


# In[46]:


# converting select to Unknown and then adding null values to this category
df['How did you hear about X Education'].replace('Select', 'Unknown', inplace = True)
df['How did you hear about X Education'] = df['How did you hear about X Education'].fillna('Unknown')


# In[47]:


print(df['How did you hear about X Education'].value_counts())
print(df['How did you hear about X Education'].isnull().sum())


# In[48]:


print(df['What is your current occupation'].value_counts())
print(df['What is your current occupation'].isnull().sum())


# In[49]:


feature_list = df.columns.to_list()
null_features = []
for feature in feature_list:    
    if ((df[feature].isnull().sum()/len(df))*100) >0:
        null_features.append(feature)
null_features


# In[50]:


# converting null values to Unknown
df['What is your current occupation'] = df['What is your current occupation'].fillna('Unknown')


# In[51]:


print(df['What is your current occupation'].value_counts())
print(df['What is your current occupation'].isnull().sum())


# In[52]:


print(df['What matters most to you in choosing a course'].value_counts())
print(df['What matters most to you in choosing a course'].isnull().sum())


# In[53]:


# we can drop this column since most of the values belong to `Better Career Prospects` and others are unknown.
df = df.drop('What matters most to you in choosing a course', 1)


# In[54]:


print(df['Tags'].value_counts())
print(df['Tags'].isnull().sum())


# In[55]:


# converting null values to Unknown
df['Tags'] = df['Tags'].fillna('Unknown')


# In[56]:


print(df['Tags'].value_counts())
print(df['Tags'].isnull().sum())


# In[57]:


print(df['City'].value_counts())
print(df['City'].isnull().sum())


# In[58]:


# converting select to Unknown and then adding null values to this category
df['City'].replace('Select', 'Unknown', inplace = True)
df['City'] = df['City'].fillna('Unknown')


# In[59]:


print(df['City'].value_counts())
print(df['City'].isnull().sum())


# In[60]:


print(df['Lead Profile'].value_counts())
print(df['Lead Profile'].isnull().sum())


# In[61]:


# converting select to Unknown and then adding null values to this category
df['Lead Profile'].replace('Select', 'Unknown', inplace = True)
df['Lead Profile'] = df['Lead Profile'].fillna('Unknown')


# In[62]:


print(df['Lead Profile'].value_counts())
print(df['Lead Profile'].isnull().sum())


# In[63]:


for feature in df.columns:
    print(feature, ' - ', df[feature].nunique())


# Dropping columns with 1 unique value since these columns won't really help our outcomes. Also, Prospect ID and Lead Number are the same number and both represent the same customer, hence, dropping the Prospect ID to reduce redundancy.

# In[64]:


df = df.drop(['I agree to pay the amount through cheque', 'Get updates on DM Content', 'Update me on Supply Chain Content', 'Receive More Updates About Our Courses', 'Magazine', 'Prospect ID'], axis=1)


# In[65]:


df.columns


# In[66]:


# checking for outliers in numerical features of the dataset
df.info()


# In[67]:


df.describe()


# In[68]:


plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
sns.boxplot(df['TotalVisits'])

plt.subplot(2,2,2)
sns.boxplot(df['Total Time Spent on Website'])

plt.subplot(2,2,3)
sns.boxplot(df['Page Views Per Visit'])


# In[69]:


# removing outliers using the IQR

Q1 = df['TotalVisits'].quantile(0.25)
Q3 = df['TotalVisits'].quantile(0.75)
IQR = Q3 - Q1
df = df.loc[(df['TotalVisits'] >= Q1 - 1.5*IQR) & (df['TotalVisits'] <= Q3 + 1.5*IQR)]

Q1 = df['Page Views Per Visit'].quantile(0.25)
Q3 = df['Page Views Per Visit'].quantile(0.75)
IQR = Q3 - Q1
df=df.loc[(df['Page Views Per Visit'] >= Q1 - 1.5*IQR) & (df['Page Views Per Visit'] <= Q3 + 1.5*IQR)]


# In[70]:


plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
sns.boxplot(df['TotalVisits'])

plt.subplot(2,2,2)
sns.boxplot(df['Total Time Spent on Website'])

plt.subplot(2,2,3)
sns.boxplot(df['Page Views Per Visit'])


# ### Mapping binary values to 1 and 0

# In[71]:


def mapping(x): # creating the mapping function
    return x.map({'Yes':1, 'No':0})

# making a list of features to be mapped to 1 and 0
features_mapping = ['Search', 'Do Not Email', 'Do Not Call', 'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations', 'A free copy of Mastering The Interview']

df[features_mapping] = df[features_mapping].apply(mapping)


# In[72]:


df.head()


# In[73]:


# creating dummy variables for the features with no Unknown values
df = pd.get_dummies(df, columns=['Lead Origin', 'Lead Source', 'Country', 'Last Notable Activity'], drop_first=True)


# In[74]:


# Creating dummmy variables for the features with 'Unknown' values and dropping the 'Unknown' created feature


# Creating dummy variables for the variable 'Last Activity'
dummy = pd.get_dummies(df['Last Activity'], prefix='Last Activity')
final_dummy = dummy.drop(['Last Activity_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)

# Creating dummy variables for the variable 'What is your current occupation'
dummy = pd.get_dummies(df['What is your current occupation'], prefix='What is your current occupation')
final_dummy = dummy.drop(['What is your current occupation_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)

# Creating dummy variables for the variable 'Lead Profile'
dummy = pd.get_dummies(df['Lead Profile'], prefix='Lead Profile')
final_dummy = dummy.drop(['Lead Profile_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)

# Creating dummy variables for the variable 'Specialization'
dummy = pd.get_dummies(df['Specialization'], prefix='Specialization')
final_dummy = dummy.drop(['Specialization_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)

# Creating dummy variables for the variable 'City'
dummy = pd.get_dummies(df['City'], prefix='City')
final_dummy = dummy.drop(['City_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)

# Creating dummy variables for the variable 'Tags'
dummy = pd.get_dummies(df['Tags'], prefix='Tags')
final_dummy = dummy.drop(['Tags_Unknown'], 1)
df = pd.concat([df,final_dummy], axis=1)


# In[75]:


df.shape


# ### Dropping the features for which dummy variables were created

# In[76]:


df = df.drop(['How did you hear about X Education', 'Last Activity', 'What is your current occupation', 'Lead Profile', 'Specialization', 'City', 'Tags'],axis=1)


# In[ ]:





# In[77]:


df.shape


# ### Splitting the data into Training and Test datasets

# In[78]:


from sklearn.model_selection import train_test_split as tts


# In[79]:


X = df.drop(['Lead Number', 'Converted'], axis = 1)
y = df['Converted']


# In[80]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = tts(X, y, train_size=0.7, test_size=0.3, random_state=100)


# Scaling all the features before proceeding further.

# In[81]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[82]:


X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


# In[83]:


X_train.head()


# ### Model Building

# In[84]:


import statsmodels.api as sm


# In[85]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=20) #selecting 20 feature for this model
rfe = rfe.fit(X_train, y_train)
list(zip(X_train, rfe.support_, rfe.ranking_))


# In[86]:


#Assigning the 20 selecting columns to a col variable.
cols = X_train.columns[rfe.support_]
cols


# In[87]:


X_train_sm = sm.add_constant(X_train[cols])
m1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = m1.fit()
res.summary()


# In[88]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:20]


# In[89]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted' : y_train.values, 'Conversion_Prob' : y_train_pred})
y_train_pred_final['LeadID'] = y_train.index
y_train_pred_final.head()


# In[90]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[91]:


from sklearn import metrics


# In[92]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion)


# In[93]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# ### Our current model is 93 percent accurate

# Checking VIFs

# In[94]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[95]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# VIF values for our features seem to be fine. Let's drop variables on the basis of p values.

# In[96]:


cols = cols.drop('Tags_number not provided', 1)
cols


# ## Model 2

# In[97]:


X_train_sm = sm.add_constant(X_train[cols])
m2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = m2.fit()
res.summary()


# In[98]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)


# In[99]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted' : y_train.values, 'Conversion_Prob' : y_train_pred})
y_train_pred_final['LeadID'] = y_train.index
y_train_pred_final.head()


# In[100]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[101]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion)


# In[102]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# The accuracy has not dropped much. Let's check for VIF values for the features now.

# In[103]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping features based on p values again since VIF is in the acceptable range, i.e. less than 3.

# In[104]:


cols = cols.drop('Tags_Not doing further education', 1)
cols


# ## Model 3

# In[105]:


X_train_sm = sm.add_constant(X_train[cols])
m3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = m3.fit()
res.summary()


# In[106]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)


# In[107]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted' : y_train.values, 'Conversion_Prob' : y_train_pred})
y_train_pred_final['LeadID'] = y_train.index
y_train_pred_final.head()


# In[108]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[109]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion)


# In[110]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# The accuracy has increased a little. Let's check for VIF values for the features now.

# In[111]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping features based on p values again since VIF is in the acceptable range, i.e. less than 3.

# In[112]:


cols = cols.drop('Tags_invalid number', 1)
cols


# ## Model 4

# In[113]:


X_train_sm = sm.add_constant(X_train[cols])
m4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = m4.fit()
res.summary()


# In[114]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)


# In[115]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted' : y_train.values, 'Conversion_Prob' : y_train_pred})
y_train_pred_final['LeadID'] = y_train.index
y_train_pred_final.head()


# In[116]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[117]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion)


# In[118]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# The accuracy has dropped a little, but nothing worrysome here. Let's check for VIF values for the features now.

# In[119]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Dropping features based on p values again since VIF is in the acceptable range, i.e. less than 3.

# In[120]:


cols = cols.drop('Tags_wrong number given', 1)
cols


# ## Model 5

# In[121]:


X_train_sm = sm.add_constant(X_train[cols])
m5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = m5.fit()
res.summary()


# In[122]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)


# In[123]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted' : y_train.values, 'Conversion_Prob' : y_train_pred})
y_train_pred_final['LeadID'] = y_train.index
y_train_pred_final.head()


# In[124]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[125]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion)


# In[126]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# The accuracy has dropped, but it is still more than 91 percent. Let's check for VIF values for the features now.

# In[127]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 1. We can see that now most of our P-values for almost all of our variables are equal to 'Zero' which indicates that these variables are statistically significant so we do not need to drop more feature variables.
# 2. Also the accuracy of our model hasn't dropped even after removing so many of the feature columns at around 91.5%

# In[128]:


# correlation matrix 
plt.figure(figsize = (20,10),dpi=600)  
sns.heatmap(X_train[cols].corr(),annot = True, cmap='coolwarm')
plt.show()

plt.savefig('corr.png')


# ### Metrics beyond simply accuracy

# In[129]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[130]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[131]:


# Let us calculate specificity
TN / float(TN+FP)


# In[132]:


# Calculate false postive rate
print(FP/ float(TN+FP))


# In[133]:


# positive predictive value 
print (TP / float(TP+FP))


# In[134]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Plotting the ROC Curve

# In[135]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (RoC) curve')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[136]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, 
                                         drop_intermediate = False )


# In[137]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# ## Finding Optimal Cutoff Point

# In[138]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head(10)


# In[139]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[140]:


# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.vlines(x=0.33, ymax=1, ymin=0, colors="r", linestyles="--")
plt.show()


# From the above curve, 0.33 seems to be the optimum point to take as the cutoff probability

# In[141]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.33 else 0)
y_train_pred_final.head()


# In[142]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[143]:


# Confusion matrix
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2


# In[144]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[145]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[146]:


# Let us calculate specificity
TN / float(TN+FP)


# In[147]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[148]:


# Positive predictive value 
print (TP / float(TP+FP))


# ## Precision and Recall

# Precision: TP / TP + FP

# In[149]:


confusion2[1,1]/(confusion2[0,1]+confusion2[1,1])


# In[150]:


from sklearn.metrics import precision_score, recall_score


# In[151]:


precision_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[152]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[153]:


from sklearn.metrics import precision_recall_curve


# In[154]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[155]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[156]:


from sklearn.metrics import classification_report


# In[157]:


print(classification_report(y_train_pred_final.Converted, y_train_pred_final.final_predicted))


# ### Making predictions on the test set

# In[158]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_test.head()


# In[159]:


X_test = X_test[cols]
X_test.head()


# In[160]:


# adding constant for statsmodel
X_test_sm = sm.add_constant(X_test)


# In[161]:


# making prediction on the test set
y_test_pred = res.predict(X_test_sm)


# In[162]:


y_pred = pd.DataFrame(y_test_pred)


# In[163]:


y_pred.head()


# In[164]:


y_test_df = pd.DataFrame(y_test)


# In[165]:


y_test_df.head()


# In[166]:


# Putting LeadID to index
y_test_df['LeadID'] = y_test_df.index
y_test_df.head()


# In[167]:


# concatenating both the prediction and the orginal labels
y_pred_final = pd.concat([y_test_df, y_pred],axis=1)


# In[168]:


y_pred_final.head()


# In[169]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Conversion_Prob'})
# Rearranging the columns
y_pred_final = y_pred_final[['LeadID','Converted','Conversion_Prob']]


# In[170]:


y_pred_final.head()


# In[171]:


y_pred_final['Predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.33 else 0)


# In[172]:


# Let's check the overall accuracy.
accuracy_score=metrics.accuracy_score(y_pred_final.Converted, y_pred_final.Predicted)
accuracy_score


# ### Confusion matrix

# In[173]:


confusion_test_set = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.Predicted)
print(confusion_test_set)


# In[174]:


TP = confusion_test_set[1,1] # true positive 
TN = confusion_test_set[0,0] # true negatives
FP = confusion_test_set[0,1] # false positives
FN = confusion_test_set[1,0] # false negatives


# <b>Sensitivity</b>

# In[175]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# <b>Specificity</b>

# In[176]:


# Let us calculate specificity
TN / float(TN+FP)


# <b>False Postive Rate</b>

# In[177]:


# Calculate false postive rate - predicting converion when customer does not have converted
print(FP/ float(TN+FP))


# <b>Positive Predictive Value</b>

# In[178]:


# Positive predictive value 
print (TP / float(TP+FP))


# <b>Negative Predicted Value</b>

# In[179]:


# Negative predictive value
print (TN / float(TN+ FN))


# <b>Precision</b>

# In[180]:


#precision
confusion_test_set[1,1]/(confusion_test_set[0,1]+confusion_test_set[1,1])


# <b>Recall</b>

# In[181]:


#recall
confusion_test_set[1,1]/(confusion_test_set[1,0]+confusion_test_set[1,1])


# <b>Classification Report</b>

# In[182]:


print(classification_report(y_pred_final.Converted, y_pred_final.Predicted))


# <b>Precision recall curve</b>

# In[183]:


p, r, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Conversion_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Plotting the ROC Curve for Test Dataset

# In[184]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()

    return fpr,tpr, thresholds


# In[185]:


fpr, tpr, thresholds = metrics.roc_curve(y_pred_final.Converted, y_pred_final.Conversion_Prob, drop_intermediate = False)


# In[186]:


draw_roc(y_pred_final.Converted, y_pred_final.Conversion_Prob)
plt.show


# Area under the ROC curve is around 0.96 which means our model seems to be doing well on the test set as well

# In[187]:


y_pred_final.head()


# In[188]:


y_pred_final['Lead Score'] = y_pred_final['Conversion_Prob']*100
y_pred_final.head()


# In[189]:


y_pred_final = pd.merge(df[['Lead Number']], y_pred_final,how='inner',left_index=True, right_index=True)


# In[190]:


y_pred_final.head()  # test dataset with all the Lead Score values


# In[191]:


y_train_pred_df = y_train_pred_final[['Converted', 'Conversion_Prob', 'LeadID','Predicted']]
y_train_pred_df.head()


# In[192]:


y_train_pred_df = pd.merge(df[['Lead Number']], y_train_pred_df,how='inner',left_index=True, right_index=True)
y_train_pred_df.head()


# In[193]:


y_train_pred_df['Lead Score'] = y_train_pred_df['Conversion_Prob']*100
y_train_pred_df.head()     # train dataset with all the Lead Score values


# Final dataframe with all the Lead Scores

# In[194]:


final_df_lead_score = pd.concat([y_train_pred_df,y_pred_final],axis=0)
final_df_lead_score.head()


# In[195]:


final_df_lead_score = final_df_lead_score.set_index('LeadID')

final_df_lead_score = final_df_lead_score[['Lead Number','Converted','Conversion_Prob','Predicted','Lead Score']]


# Final dataframe with the Lead Scores for all the LeadID

# In[196]:


final_df_lead_score.head()  # final dataframe with all the Lead Scores


# In[197]:


final_df_lead_score.shape


# Determining Feature Importance of our final model

# In[198]:


# coefficients of our final model 

pd.options.display.float_format = '{:.2f}'.format
new_params = res.params[1:]
new_params


# In[199]:


# Getting a relative coeffient value for all the features wrt the feature with the highest coefficient

feature_importance = new_params
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance


# In[200]:


# Sorting the feature variables based on their relative coefficient values

sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')


# In[201]:


feature_importance_df = pd.DataFrame(feature_importance).reset_index().sort_values(by=0,ascending=False)
feature_importance_df = feature_importance_df.rename(columns={'index':'Variables', 0:'Relative coeffient value'})
feature_importance_df = feature_importance_df.reset_index(drop=True)
feature_importance_df.head(5)


# The top 5 variables are:<br>
# 1. Tags_Closed by Horizzon<br>
# 2. Tags_Lost to EINS<br>
# 3. Lead Source_Welingak Website<br>
# 4. Tags_Will revert after reading the email<br>
# 5. Last Activity_SMS Sent
