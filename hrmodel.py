import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('turnover.csv')
# Change the type of the "salary" column to categorical
data.salary = data.salary.astype('category')

# Provide the correct order of categories
data.salary = data.salary.cat.reorder_categories(['low', 'medium', 'high'])

# Encode categories
data.salary = data.salary.cat.codes

departments = pd.get_dummies(data.department)
departments = departments.drop("accounting", axis=1)
data = data.drop("department", axis=1)
data = data.join(departments)
target = data["churn"]
features = data.drop("churn",axis=1)
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)
depth = [i for i in range(2,10)]

samples = [i for i in range(20,100,10)]
parameters = dict(max_depth=depth,min_samples_leaf=samples)
model=DecisionTreeClassifier(random_state=42, class_weight="balanced")

# initialize the param_search function using default model and parameters above
param_search = GridSearchCV(model, parameters ,cv=5)
param_search.fit(features_train,target_train)
print(param_search.best_params_)

bestdepth=param_search.best_params_['max_depth']
minsampleleaves=param_search.best_params_['min_samples_leaf']

model_best = DecisionTreeClassifier(random_state=42, class_weight="balanced" , max_depth=bestdepth ,min_samples_leaf =minsampleleaves)
model_best.fit(features_train,target_train)


# Calculate feature importances
feature_importances = model_best.feature_importances_
# Create a list of features__
feature_list = list(features)
# Save the results inside a DataFrame
relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])
# Sort the DataFrame to learn most important features
relative_importances.sort_values(by="importance", ascending=False)
# select only features with relative importance higher than 1%
selected_features = relative_importances[relative_importances.importance>0.01]

selected_list = selected_features.index
# transform both features_train and featu+res_test components to include only selected features
features_train_selected = features_train[selected_list]
features_test_selected = features_test[selected_list]

model_best.fit(features_train_selected,target_train)

prediction_best = model_best.predict(features_test_selected)
#accuracy
print(model_best.score(features_test_selected,target_test)*100)
#roc/auc curve area 
print(roc_auc_score(prediction_best,target_test)*100)

data = data.rename(columns={
                        
                        'number_of_projects': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        
                        'churn' : 'turnover'
                        })
#heatmap
corr = data.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
corr

#projectCount V.S. turnover
clarity_color_table = pd.crosstab(index=data["projectCount"], 
                          columns=data["turnover"])

clarity_color_table.plot(kind="bar", 
                 figsize=(5,5),
                 stacked=True)

#KDEPlot: Kernel Density Estimate Plot for average Monthly Hours
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(data.loc[(data['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(data.loc[(data['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')
plt.title('Average monthly hours worked' )

#KDEPlot: Kernel Density Estimate Plot for Evaluation
fig2 = plt.figure(figsize=(10,4),)
ax=sns.kdeplot(data.loc[(data['turnover'] == 0),'evaluation'] , color='b',shade=True,label='no turnover')
ax=sns.kdeplot(data.loc[(data['turnover'] == 1),'evaluation'] , color='r',shade=True, label='turnover')
plt.title('Last evaluation')



