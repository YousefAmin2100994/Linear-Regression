#!/usr/bin/env python
# coding: utf-8

# ## *Importing Modules*

# In[59]:


import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,SGDRegressor,Lasso,Ridge
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# ## *analysis for data set*

# In[60]:


data_frame=pd.read_csv(r'H:\insurance.csv')
data_frame.head()


# In[61]:


data_frame.rename(columns={'bmi':'Relative Mass','charges':'medical costs'},inplace=True)
data_frame.head()


# In[62]:


columns_list=data_frame.columns
columns_list


# ### *1.Feature Extraction from age*

# In[63]:


data_frame['age'].describe()


# In[64]:


def knowing_range_age(age):
    if age < 27:
        return 'Young adult'
    elif age < 55:
        return 'Middle Age adult'
    else:
        return 'Old adult'
data_frame['Age_range']=data_frame['age'].apply(knowing_range_age)
data_frame.head()


# ### *Encoding for Age_range* 

# In[65]:


def encoding_age(age):
    if age < 27:
        return 1
    elif age < 55:
        return 2
    else:
        return 3
data_frame['Age_range']=data_frame['age'].apply(encoding_age)
data_frame.head()


# ### *Encoding for sex*

# In[66]:


# Using make_column_transformer to One-Hot Encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

transformer = make_column_transformer(
    (OneHotEncoder(), ['sex']),
    remainder='passthrough')

transformed = transformer.fit_transform(data_frame)

# Update the DataFrame after transformation
data_frame = pd.DataFrame(
    transformed, 
    
)
data_frame.head()


# In[67]:


data_frame.rename(columns={0:'Female',1:'Male',2:'age',3:'Relative Mass',4:'children',5:'smoker',7:"medical costs",8:'Age_range'},inplace=True)
data_frame.head()


# In[68]:


data_frame.drop(6,axis=1,inplace=True)
data_frame.head()


# ### *Encoding for smoker*

# In[69]:


data_frame['smoker']=data_frame['smoker'].apply(lambda x : 1 if x=='yes' else 0)
data_frame.head()


# ## *Feature Selection*

# In[70]:


columns_list=data_frame.columns
columns_list


# In[71]:


from sklearn.feature_selection import SelectKBest, f_regression,f_classif

# Assuming 'data_frame' is your DataFrame
X = data_frame[['Female', 'Male', 'age', 'Relative Mass', 'children', 'smoker', 'Age_range']]
y = data_frame['medical costs']

# Initialize the selector
selector = SelectKBest(score_func=f_classif, k=5)

# Fit and transform the data
X_selected = selector.fit_transform(X, y)

# Get the selected features
selected_features = X.columns[selector.get_support()]

# Create a new DataFrame with selected features
X = pd.DataFrame(X_selected, columns=selected_features)
X.head()


# In[72]:


data_frame=pd.concat([X,data_frame['medical costs']],axis=1)
data_frame.head()


# ### *Handling Duplicated Values*

# In[73]:


data_frame.duplicated().sum()


# In[74]:


data_frame.drop_duplicates(inplace=True)
data_frame.head()


# In[75]:


data_frame.reset_index(inplace=True)
data_frame.drop('index',axis=1,inplace=True)
data_frame.head()


# In[76]:


data_frame.duplicated().sum() # :)


# ### *Handling Null Values*

# In[77]:


data_frame.isnull().sum()


# In[79]:


# May be null values written as zeros :)
data_frame[data_frame['age']==0]


# In[80]:


data_frame[data_frame['Relative Mass']==0]


# In[81]:


# so there arenot null values :)


# ### *Handling outliers*

# In[84]:


plt.style.use('ggplot')
data_frame['age'].plot(kind='box')
plt.title('Summary of data') #so there isnot any outliers


# In[85]:


data_frame['Relative Mass'].plot(kind='box')
plt.title('Summary of Relative Mass') #so there isnot any outliers


# In[87]:


#there are outliers in Relative mass so count number of them first
def percetage_of_outliers_relative_mass(data):
    all_num=0
    count=0
    for i in data:
        all_num+=1
        if i > 47:
            count+=1
    return count,all_num,count/all_num
percetage_of_outliers_relative_mass(data_frame['Relative Mass']) #very small percantage 


# In[88]:


data_frame=data_frame[data_frame['Relative Mass']<47]
data_frame.head()


# In[90]:


data_frame.shape


# In[97]:


data_frame.head()


# In[98]:


data_frame.drop(['index','level_0'],axis=1,inplace=True)
data_frame.head()


# In[99]:


data_frame['Relative Mass'].plot(kind='box')
plt.title('Summary of Relative Mass') #nice 


# In[100]:


data_frame['children'].plot(kind='box')
plt.title('Summary of Children') #so there isnot any outliers


# In[101]:


data_frame['medical costs'].plot(kind='box')
plt.title('Summary of Medical costs') 


# In[114]:


def percetage_of_outliers_for_out(data):
    all_num=0
    count=0
    for i in data:
        all_num+=1
        if i > 25000:
            count+=1
    return count,all_num,count/all_num
percetage_of_outliers_for_out(data_frame['medical costs'])


# In[115]:


# because its very important feature so we must delete rows not substitute
data_frame=data_frame[data_frame['medical costs']<25000]
data_frame.head()


# In[116]:


data_frame.reset_index(inplace=True)
data_frame.drop('index',axis=1,inplace=True)
data_frame.head()


# In[117]:


data_frame.shape


# In[118]:


data_frame['medical costs'].plot(kind='box')
plt.title('Summary of Medical costs') 


# ## *Making visualization* 

# In[119]:


data_frame.head()


# In[121]:


sns.heatmap(data_frame.corr(),annot=True)
plt.title('Heat map for showing correlation')


# In[122]:


sns.kdeplot(data_frame['medical costs'])


# In[125]:


# Assuming data_frame is your DataFrame
plt.figure(figsize=(12, 8))

# Plot for 'children'
plt.subplot(2, 2, 1)
sns.kdeplot(data_frame['children'], color='blue')
plt.title('Distribution of Children')
plt.xlabel('Number of Children')
plt.ylabel('Density')

# Plot for 'Relative Mass'
plt.subplot(2, 2, 2)
sns.kdeplot(data_frame['Relative Mass'], color='green')
plt.title('Distribution of Relative Mass')
plt.xlabel('Relative Mass')
plt.ylabel('Density')

# Plot for 'age'
plt.subplot(2, 2, 3)
sns.kdeplot(data_frame['age'], color='orange')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Density')

plt.tight_layout()
plt.show()


# In[133]:


data_frame['medical costs'].plot(kind='hist',alpha=0.87)
plt.xlabel('Medical costs')
plt.title('Histogram for Medical Costs')


# In[134]:


data_frame['children'].plot(kind='hist',alpha=0.87)
plt.xlabel('children')
plt.title('Histogram for children')


# In[135]:


data_frame['Relative Mass'].plot(kind='hist',alpha=0.87)
plt.xlabel('Relative Mass')
plt.title('Histogram for Relative Mass')


# In[138]:


data=data_frame.corr()['medical costs']
data


# In[153]:


names=list(data.index[:-1])
values=list(data.values[:-1])
names.remove('Relative Mass')
values.pop(1)
names,values


# In[292]:


# Explode the first slice
explode = (0.05, 0, 0.066, 0.025)

# Choose colors for each feature
colors = sns.color_palette("Blues", 5)

# Set up the figure with subplots
plt.figure(figsize=(10, 6))

# Plot the pie chart
plt.subplot(1, 2, 1)
plt.pie(values, labels=names, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
plt.title('Features Effect on Medical Costs by Pie chart')

# Plot bar chart
plt.subplot(1, 2, 2)
bars = plt.bar(names, values, edgecolor='k', alpha=0.88)

# Text on bar chart
for xx,yy in zip(names,values):
    plt.text(xx,yy,np.round(yy,1), ha='center', va='bottom', fontsize=10)

plt.title('Features Effect on Medical Costs by Bar chart')

# Display the chart
plt.show()


# ## *Important to know*

# In[166]:


data_frame.describe()


# In[167]:


data_frame.info()


# In[174]:


columns_list=data_frame.columns
columns_list


# In[173]:


x_int=['age','children','Age_range','smoker']
for col in x_int:
    data_frame[col]=data_frame[col].astype(int)
for i in ['Relative Mass','medical costs']:
    data_frame[col]=data_frame[col].astype(float)
data_frame.info()


# ## *Making Multiple linear regression model*

# In[224]:


x=data_frame[['age', 'Relative Mass', 'children', 'smoker', 'Age_range']]
y=data_frame['medical costs']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=107)
x_train.shape,x_test.shape


# In[225]:


y_train.shape,y_test.shape


# In[226]:


model=LinearRegression()
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
y_train_pred


# In[227]:


y_test_pred


# ## *Evaluation of the model for train data* 

# In[228]:


mean_squared_error(y_train,y_train_pred)


# In[229]:


mean_absolute_error(y_train,y_train_pred)


# In[230]:


r2_score(y_train,y_train_pred)


# In[231]:


model.score(x_train,y_train)


# In[232]:


plt.scatter(y_train,y_train_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation')


# In[234]:


sns.kdeplot(y_train_pred-y_train)
plt.xlabel('Residual graph')


# ## *Evaluation of the model for test data* 

# In[235]:


mean_squared_error(y_test,y_test_pred)


# In[238]:


mean_absolute_error(y_test,y_test_pred)


# In[239]:


r2_score(y_test,y_test_pred)


# In[241]:


plt.scatter(y_test,y_test_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation"(test)"')


# In[242]:


sns.kdeplot(y_test_pred-y_test)
plt.xlabel('Residual graph')


# ## *Lasso model for solving overfitting*

# In[243]:


#there is overfitting problem so try to use lasso
model=Lasso()
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
y_train_pred


# ## *Evaluation of the model train*

# In[257]:


mean_squared_error(y_train,y_train_pred)


# In[247]:


mean_absolute_error(y_train,y_train_pred)


# In[248]:


model.score(x_train,y_train)


# In[255]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(y_train,y_train_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation')
plt.subplot(1,2,2)
sns.kdeplot(y_train_pred-y_train)
plt.xlabel('Residual graph')


# ## *Evaluation of test data*

# In[258]:


mean_squared_error(y_test,y_test_pred)


# In[259]:


mean_absolute_error(y_test,y_test_pred)


# In[260]:


r2_score(y_test,y_test_pred)


# In[272]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(y_test,y_test_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation')
plt.subplot(1,2,2)
sns.kdeplot(y_test_pred-y_test)
plt.xlabel('Residual graph')
plt.title('Kernal graph of Residuals')


# ## *Ridge Model*

# In[262]:


model=Ridge()
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
y_train_pred


# In[263]:


y_test_pred


# ## *Evaluation of model for train data*

# In[264]:


mean_squared_error(y_train,y_train_pred)


# In[267]:


mean_absolute_error(y_train,y_train_pred)


# In[268]:


r2_score(y_train,y_train_pred)


# In[270]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(y_train,y_train_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation')
plt.subplot(1,2,2)
sns.kdeplot(y_train_pred-y_train)
plt.xlabel('Residual graph')
plt.title('Kernal graph of Residuals')


# ## *Evaluation of model for test data*

# In[273]:


mean_squared_error(y_test,y_test_pred)


# In[274]:


mean_absolute_error(y_test,y_test_pred)


# In[275]:


r2_score(y_test,y_test_pred)


# In[276]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(y_test,y_test_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation')
plt.subplot(1,2,2)
sns.kdeplot(y_test_pred-y_test)
plt.xlabel('Residual graph')
plt.title('Kernal graph of Residuals')


# ## *Polynomial Regression*

# In[293]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

param_grid = {'polynomialfeatures__degree': np.arange(10),}
poly_grid = GridSearchCV(PolynomialRegression(), param_grid, cv=10, scoring='neg_mean_squared_error', refit=True)


# In[294]:


poly_grid.fit(x_train,y_train)


# In[295]:


print(poly_grid.best_params_)
print(poly_grid.best_score_)


# In[301]:


poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x_train)
model=LinearRegression()
model.fit(poly_features,y_train)
y_pred=model.predict(poly_features)
y_pred


# ## *Evaluation for polynomial Regression*

# In[302]:


mean_absolute_error(y_train,y_pred)


# In[304]:


mean_squared_error(y_train,y_pred)


# In[305]:


r2_score(y_train,y_pred)


# In[306]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(y_train,y_train_pred,c='g')
plt.xlabel('original y')
plt.ylabel('predicted y')
plt.title('scatter plot to show the correlation')
plt.subplot(1,2,2)
sns.kdeplot(y_train_pred-y_train)
plt.xlabel('Residual graph')
plt.title('Kernal graph of Residuals')


# ## *so the polynomial with degree 2 is the best :)*
# ## *Happy end of assignment :)*

# ## *joe Amin*

# In[ ]:




