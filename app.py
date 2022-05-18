#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:30:15 2022




"""
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle 


                                    ######## Loading the data  ##########


#url = 'https://opendata.arcgis.com/api/v3/datasets/d2d77227e7d34a0089ab43bc635b948a_0/downloads/data?format=csv&spatialRefId=26717'

df_bicycle= pd.read_csv('/Users/abrarmohammed/Documents/Bicycle_Thefts.csv')


                                    ######## Data exploration  ##########

df_bicycle.shape
df_bicycle.head(9)
df_bicycle.info()
df_bicycle.describe()
df_bicycle.keys()
df_bicycle['Status'].unique()
df_bicycle['Status'].value_counts()
df_bicycle.isnull().sum()
#df_bicycle.isnull().mean()

for col in df_bicycle.columns:
    print(col, ':' , len(df_bicycle[col].unique()), 'lables')




                                        ########    Plot Graphs  ##########
                                    
                                    
# Hist Graph                               
df_bicycle.hist(bins=50, figsize=(20,15))
df_bicycle["Status"].hist()


df_bicycle['Premises_Type'].value_counts()
df_bicycle['Premises_Type'].value_counts().plot.bar()


df_bicycle['Bike_Type'].value_counts()
df_bicycle['Bike_Type'].value_counts().plot.bar()


df_bicycle['Occurrence_Year'].value_counts()
df_bicycle['Occurrence_Year'].value_counts().plot.bar()

plt.figure()
df_bicycle.Occurrence_Year.plot(kind = 'hist')


#pie charts
#plt.style.use('ggplot')
Stolen = df_bicycle[df_bicycle['Status'] == 'STOLEN'].count()[0]
Unknown = df_bicycle[df_bicycle['Status'] == 'UNKNOWN'].count()[0]    
Recovered = df_bicycle[df_bicycle['Status'] == 'RECOVERED'].count()[0]  
data =  [Stolen, Unknown, Recovered] 
labels =['Stolen','Unknown','Recovered']
colors = ['#ff9999','#66b3ff','#99ff99' ]
explode=(0,0.8,0.8)
#plt.pie([Stolen, Unknown, Recovered], labels = labels, autopct='%.2f %%', pctdistance=0.7)

#plt.pie([Stolen, Unknown, Recovered], labels = labels, autopct='%.2f %%', pctdistance=0.5, colors=colors,explode=explode, shadow=True)
plt.pie(data, labels = labels, autopct='%.2f %%', pctdistance=0.5, colors=colors,explode=explode, shadow=True)
plt.tight_layout()
plt.title('Status of Bicycle Theft')    
plt.show()

'''
plt.style.use('ggplot')
plt.pie([Stolen, Unknown, Recovered], labels = labels, autopct='%.2f %%', pctdistance=0.7)
plt.title('Status of Bicycle Theft')    
plt.show()
'''


#Occurrence_DayOfWeek
plt.style.use('ggplot')
Sun = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Sunday'].count()[0]
Mon = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Monday'].count()[0] 
Tue = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Tuesday'].count()[0] 
Wed = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Wednesday'].count()[0] 
Thurs = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Thurs'].count()[0] 
Fri = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Fri'].count()[0] 
Sat = df_bicycle[df_bicycle['Occurrence_DayOfWeek'] == 'Sat'].count()[0] 

data = [Sun, Mon, Tue, Wed,Thurs, Fri, Sat]
labels =['Sun','Mon','Tue','Wed','Thurs','Fri','Sat']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#aabbcc', '#abcdef', '#ffcc99' ]
explode=(0,0,0,0.1,0, 0.5,0.5)
circle = plt.Circle((0, 0), 0.7, color='white') 

    
plt.pie(data, autopct='%.1f %%', pctdistance=0.6, colors=colors,explode=explode, shadow=True, startangle=300 )


p = plt.gcf()   
p.gca().add_artist(circle)

plt.title('Days of Bicycle Theft')  
plt.legend(labels, loc="upper right")
plt.axis('equal')  
plt.tight_layout()  
plt.show()

'''
plt.pie([Sun, Mon, Tue, Wed,Thurs,Fri,Sat], labels = labels, autopct='%.2f %%', pctdistance=0.3, explode=explode)
plt.title('Days of Bicycle Theft')    
plt.show()

plt.style.use('ggplot')

'''


# 
pd.crosstab(df_bicycle.Bike_Type, df_bicycle.Status,).plot(kind='bar')
plt.title('Status by Bike_Type')
plt.xlabel('Bike Type')
plt.ylabel('Status')

pd.crosstab(df_bicycle.Occurrence_Month, df_bicycle.Status,).plot(kind='bar')
plt.title('Status by Occurrence_Month')
plt.xlabel('Occurrence Month')
plt.ylabel('Status')




sns.catplot(x="Status", y="Cost_of_Bike", data=df_bicycle)

df_bicycle['NeighbourhoodName'].value_counts().head(10).plot.barh(title="Top 10 Neighbourhoods with most number of Bicycle Theft")




                                    ######## Missing data evaluations  #########



# checking the null values mean and sum
df_bicycle.isnull().mean()


#dropping the bike model
df_bicycle.drop(['Bike_Model'], axis=1, inplace=True)

#checking columns with most null values
df_bicycle['Bike_Colour'].unique()
df_bicycle['Cost_of_Bike'].unique()



# Fill the null columns
df_bicycle = df_bicycle.fillna({
        
        'Bike_Colour' : df_bicycle['Bike_Colour'].ffill(axis=0),
        'Cost_of_Bike' : df_bicycle['Cost_of_Bike'].mean(),
        'Y' : df_bicycle['Y'].mean(),
        'X' : df_bicycle['X'].mean()
    })


# Checking again
df_bicycle.isnull().sum()




                                ######## Oversampling to handle imbalanced data  #########


# Checking the target value data
df_bicycle['Status'].value_counts()

df_bicycle['Status'] = [1 if b =='RECOVERED' else 0 for b in df_bicycle['Status']]

df_bicycle['Status'].value_counts()


# Specifying majority and minority of the data
df_majority = df_bicycle[df_bicycle.Status==0]
df_minority = df_bicycle[df_bicycle.Status==1]
 
# Upsample minority class
from sklearn.utils import resample
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=25590,   # to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
df_bicycle = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_bicycle.Status.value_counts()


                                ######## Split dataset into fetaure and target variables #########


features_data = df_bicycle.drop(columns=['Status'])
target_data = df_bicycle['Status']
print(target_data.value_counts())



   

                           ######## Feature selection visualization based on Pearson corelation #########

plt.figure(figsize=(15,10))
cor = features_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

def correlation(features_data, threshold):
    col_corr = set()                                            # Set of all the names of correlated columns
    corr_matrix = features_data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:         # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]                # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_features = correlation(features_data, 0.85)
len(set(corr_features))
corr_features


# Dropping columns based on their correlation
features_data.drop(['Latitude','Longitude','OBJECTID','OBJECTID_1','Report_DayOfYear','Report_Year'], axis=1, inplace=True)
features_data.shape





                               ######## Converting categorical columns into numeric #########


# Checkin the categorical and numericl columns
cat_features = features_data.select_dtypes('object')
numeric_features = features_data.select_dtypes('number')

print(numeric_features.columns)
print(cat_features.columns)


# Using Lable Encoder
le = LabelEncoder()

for col in cat_features:
    features_data[col] = le.fit_transform(features_data[col].astype(str))


                                       ######## MinMax Scaling #########


from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
features_data_minmax=pd.DataFrame(min_max.fit_transform(features_data),columns=features_data.columns)



                                ######## More exploration in feature data  ##########


#using Variance threshold
from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold = 0)
var_thres.fit(features_data_minmax)
var_thres.get_support()

### Finding non constant features
sum(var_thres.get_support())

# Lets Find non-constant features 
len(features_data_minmax.columns[var_thres.get_support()])
constant_columns = [column for column in features_data_minmax.columns
                    if column not in features_data_minmax.columns[var_thres.get_support()]]

print(len(constant_columns))
for column in constant_columns:
    print(column)


# determine the mutual information
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(features_data_minmax, target_data)
mutual_info
mutual_info = pd.Series(mutual_info)
mutual_info.index = features_data_minmax.columns
mutual_info.sort_values(ascending=False)


#let's plot the ordered mutual_info values per feature
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 20))



'''from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()
model.fit(X_train,y_train)
print(model.feature_importances_)

ranked_features=pd.Series(model.feature_importances_,index=X_train.columns)
print(ranked_features)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()'''

                                                           
#Dropping columns based on Mutual Info gain - <0.00
features_data_minmax.drop(columns=['Occurrence_Month','Occurrence_DayOfWeek','Report_Month','Report_DayOfWeek','Bike_Type','City','event_unique_id'], axis=1, inplace=True)



                                        ######## Train Test split  ##########


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(features_data_minmax, target_data, 
                                                 test_size=0.2, random_state=42)


                                         ########  Modeling  ##########
                                         


  
                    #################  Tahera - Decision Tree Classifier  ################
                    
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
bicycle_tahera = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=91)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
bicycle_tahera.fit(X_train,y_train)

cross_val = KFold(n_splits=5, shuffle=True, random_state=91)
scores = cross_val_score(bicycle_tahera, X_train, y_train,
                         scoring='accuracy', cv=cross_val, n_jobs=1)
scores_mean=scores.mean()
print('The five scores are \n', scores)
print('The mean score is \n', scores_mean)
#print(scores.std())

#Print out two accuracy score one for the model on the training set and testing set


y_test_pred = bicycle_tahera.predict(X_test)

accuracyScore = accuracy_score(y_test,y_test_pred)


print('The confusion matrix using testing data is:\n' ,confusion_matrix(y_test,y_test_pred))

print('Accuracy: \n', accuracyScore)
print('Classification Report: \n', classification_report(y_test, y_test_pred))

# fine tune your model using Randomized grid search
from sklearn.model_selection import GridSearchCV

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
    

tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search_cv = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search_cv.fit(X_train, y_train)
#Grid Results

best_param=grid_search_cv.best_params_
best_score=grid_search_cv.best_score_
best_estimator=grid_search_cv.best_estimator_
print('Best Parameters:\n', best_param)
print('Best score:\n',best_score)
print('Best Estimator:\n', best_estimator)

#Fitting test data
best_estimator_model = grid_search_cv.best_estimator_
y_grid_predict = best_estimator_model.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_grid_predict)
dec_accuracy = accuracy_score(y_test, y_grid_predict)


print('The confusion matrix using testing data is:\n', confusion_matrix)
print('Accuracy using Decision Tree Clasifier:', dec_accuracy)
print('Classification Report: \n', classification_report(y_test, y_test_pred))
    



    
                    #################  Hasnain - KNeighbors Classifier ################

       
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

#cross validate 3-folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring="accuracy")
print ('All scores:', scores)
print  ('Average score:', scores.mean())

kn_accuracy = scores.mean()


knn.fit(X_train,y_train)
pred = knn.predict(X_test)

sns.pairplot(df_bicycle, hue="Status")

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

error_rate = []
'''
# Will take some time
for i in range(1, 100, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    print('with K {}', i)
    print('Model accuracy score  : {0:0.4f}'.format(accuracy_score(y_test, pred_i)))
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1, 100, 10),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
    
'''


'''

                    #################  Ayo - Logistic regression Classifier ################

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics

lr = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=39)
lr.fit(X_train, y_train)
y_test_predict = lr.predict(X_test)
lr_accuracy=accuracy_score(y_test,y_test_predict)
scores = cross_val_score(lr, X_train, y_train, cv=3, scoring="accuracy")
average_score = scores.mean()
cf_matrix = (confusion_matrix(y_test, y_test_predict))


# Accuracy 

print('scores is:', scores)
print  ('Average score:', scores.mean())
print('Accuracy is:', lr_accuracy)
print("\nAccuracy: %d%%" % ((lr_accuracy(y_test, y_test_predict)) * 100))
print ('\nAll test prediction:', y_test_predict)

# Confusion Matrix
labels = y_test.unique()
print("\nConfusion Matrix")
print(cf_matrix)

# Visualize Confusion Matrix
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')


print(classification_report(y_test,y_test_predict))

# Show Coeficient and Intercept
print('\nCoefficient Values:', lr.coef_)
print('\nIntercept Value:', lr.intercept_)

# Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false
y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot([0,1],[0,1],color="red",linestyle="--")
plt.plot(fpr,tpr,label="Area under the ROC Curve="+str(auc))
plt.legend(loc=4)
plt.xlabel("False Positive Rate - FPR")
plt.ylabel("True Positive Rate - TPR ")
plt.title("Receiver Operating Characteristics - ROC Curve")
plt.text(0.6,0.5,"Baseline")
plt.text(0.3,0.8,"ROC Curve")
plt.show()


'''
                                  #################  Reihaneh - SVM  ################

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC 

'''
for kernel in ("linear", "poly", "rbf", "sigmoid"):
    
    clf_reihaneh1 = SVC(kernel= kernel)
    clf_reihaneh1.fit(X_train, y_train)
    y_pred = clf_reihaneh1.predict(X_test)
    print ("Accuracy is : " , kernel, accuracy_score(y_test,y_pred))
    
'''
 
# rbf is the best Kernel
clf_reihaneh = SVC(kernel='rbf')
clf_reihaneh.fit(X_train, y_train)
y_pred_SVC = clf_reihaneh.predict(X_test)
accuracy_rbf = accuracy_score(y_test,y_pred_SVC)
confu_matrix = confusion_matrix(y_test, y_pred_SVC)
print ("Accuracy for rbf kernel is : " , accuracy_rbf)
print('The confusion matrix for SVC model is :\n' , confu_matrix)





                 #################  Abrar - Random Forest Classifier ################

from sklearn.ensemble import RandomForestClassifier
clf_abrar=RandomForestClassifier(n_estimators=2)
clf_abrar.fit(X_train,y_train)

file = joblib.dump(clf_abrar, "RF_abrar.pkl") 
y_test_pred=clf_abrar.predict(X_test)
ran_accuracyScore = accuracy_score(y_test,y_test_pred)
#ran_confusion_matrix = confusion_matrix(y_test,y_test_pred)
ran_classification_report = classification_report(y_test, y_test_pred)


#print('The confusion matrix using testing data is:\n' ,confusion_matrix(y_test,y_test_pred))

print('Accuracy using Random Forest Classifier: \n', ran_accuracyScore)
print('Classification Report: \n', classification_report(y_test, y_test_pred))




                    #################  Bebin - MLP Classifier (Neural Network) ################


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

X_train, X_test, Y_train, y_test= train_test_split(features_data_minmax, target_data, 
                                                 test_size=0.2, random_state=42)

#Accuracy
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#Initializing the MLPClassifier
classifier_Bebin = MLPClassifier(hidden_layer_sizes=(100,50,30), max_iter=300, activation = 'relu', solver='adam', random_state = 42, verbose = True)

#Fitting the training data to the network
classifier_Bebin.fit(X_train, Y_train.astype(str))

#Predicting y for X_val
y_pred = classifier_Bebin.predict(X_test)

#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_test.astype(str))

#Printing the accuracy
print("Accuracy of MLPClassifier", accuracy(cm))

mlp_accuracy = accuracy(cm)


                    #################  Model Deployment using Flask ################



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class BicycleTheftPrediction :

    def predictRandomForest(self, features):
        clf_abrar = joblib.load("RF_abrar.pkl")
        return clf_abrar.predict(self.transformData(pd.DataFrame(features)))

    def predictDecisionTree(self, features):
        tahera_decisiontree = joblib.load("tahera_decisiontree_model.pkl")
        return tahera_decisiontree.predict(self.transformData(pd.DataFrame(features)))

    def predictKNN(self, features):
        hasnain_knn = joblib.load("KNN_Hasnain.pkl")
        return hasnain_knn.predict(self.transformData(pd.DataFrame(features)))

    def predictSVM(self, features):
        reihaneh_svm = joblib.load("SVM_reihaneh.pkl")
        return reihaneh_svm.predict(self.transformData(pd.DataFrame(features)))

    def predictLogistic(self, features):
        lrModel = joblib.load("lr_model.pkl")
        return lrModel.predict(self.transformData(pd.DataFrame(features)))
    
    def predictMLP(self, features):
        mlp_Model = joblib.load("MLP_bebin.pkl")
        return mlp_Model.predict(self.transformData(pd.DataFrame(features)))
    
    def transformData(self, features_data):
        print("Feature Data ", features_data)
        # Dropping columns based on their correlation
        '''
            columns = joblib.load("bicycle_columns.pkl")
            query = pd.get_dummies(features_data)
            features_data = query.reindex(columns=columns, fill_value=0)
        

        features_data = features_data.fillna({
            'Bike_Colour': features_data['Bike_Colour'].ffill(axis=0),
            'Cost_of_Bike': features_data['Cost_of_Bike'].mean(),
            'Y': features_data['Y'].mean(),
            'X': features_data['X'].mean()
        })
        '''
        
        #features_data.drop(['Bike_Model'], axis=1, inplace=True)
        #features_data.drop(['Latitude', 'Longitude', 'OBJECTID', 'OBJECTID_1', 'Report_DayOfYear', 'Report_Year'],
                           #axis=1, inplace=True)

        ######## Converting categorical columns into numeric #########

        # Checkin the categorical and numericl columns
        cat_features = features_data.select_dtypes('object')
        numeric_features = features_data.select_dtypes('number')

        print(numeric_features.columns)
        print(cat_features.columns)

        # Using Lable Encoder

        for col in cat_features:
            le = joblib.load("labelTransfoermers/"+col+"col.pkl")
            features_data[col] = le.transform(features_data[col].astype(str))
            print("asdfasdfasfadsf {}", features_data[col])
            ######## MinMax Scaling #########
        from sklearn.preprocessing import MinMaxScaler
        min_max = joblib.load("minMaxdataScaler.pkl")
        features_data_minmax = pd.DataFrame(min_max.transform(features_data), columns=features_data.columns)
        print("features_data_minmax",features_data_minmax)
        ######## More exploration in feature data  ##########

        import json
        result = features_data_minmax.to_json(orient="split")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        print("pardsed :- ",parsed)
        # Dropping columns based on Mutual Info gain - <0.00
        features_data_minmax.drop(
            columns=['Occurrence_Month', 'Occurrence_DayOfWeek', 'Report_Month', 'Report_DayOfWeek', 'Bike_Type',
                     'City', 'event_unique_id'], axis=1, inplace=True)
        print("asdfsafs {}",features_data_minmax.to_json)
        return features_data_minmax

from flask import Flask, request
from flask import jsonify


class WebApp:
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    bicycleTheftPrediction = BicycleTheftPrediction()
    #predict route
    @app.route('/predict', methods=["POST","GET"])
    def predict():
        if request.method == 'POST':
            req = request.get_json()
            print(req)
            if req.get('model')!= None and req.get('model') == 'RandomForest' :
                bicycleTheftPrediction = BicycleTheftPrediction()
                predictData = bicycleTheftPrediction.predictRandomForest(req.get('data')).tolist()
                print(predictData)
                return jsonify({'prediction': predictData}, {'Accuracy Score': ran_accuracyScore}, {'Classification Report': ran_classification_report })

            if req.get('model')!= None and req.get('model') == 'logistic' :
                bicycleTheftPrediction = BicycleTheftPrediction()
                predictData = bicycleTheftPrediction.predictLogistic(req.get('data')).tolist()
                print(predictData)
                return jsonify({'prediction': predictData})

            if req.get('model')!= None and req.get('model') == 'DecisionTree' :
                bicycleTheftPrediction = BicycleTheftPrediction()
                predictData = bicycleTheftPrediction.predictDecisionTree(req.get('data')).tolist()
                print(predictData)
                return jsonify({'prediction': predictData}, {'Accuracy Score': dec_accuracy})

            if req.get('model')!= None and req.get('model') == 'knn' :
                bicycleTheftPrediction = BicycleTheftPrediction()
                predictData = bicycleTheftPrediction.predictKNN(req.get('data')).tolist()
                print(predictData)
                return jsonify({'prediction': predictData},{'Accuracy Score': kn_accuracy})

            if req.get('model')!= None and req.get('model') == 'MLP' :
                bicycleTheftPrediction = BicycleTheftPrediction()
                predictData = bicycleTheftPrediction.predictMLP(req.get('data')).tolist()
                print(predictData)
                return jsonify({'prediction': predictData}, {'Accuracy Score': mlp_accuracy})

            if req.get('model')!= None and req.get('model') == 'SVM':
                bicycleTheftPrediction = BicycleTheftPrediction()
                predictData = bicycleTheftPrediction.predictSVM(req.get('data')).tolist()
                print(predictData)
                return jsonify({'prediction': predictData}, {'Accuracy Score': accuracy_rbf})
            return req

if __name__ == '__main__':
    predictionApp = WebApp()
    predictionApp.app.run(port = 8080,debug=True)

