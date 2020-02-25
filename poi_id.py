import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.regression.linear_model import OLS
from tester import test_classifier
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
SEED = 5678
np.random.seed(SEED)

"""
===============================================================================

StudentizedOLSClassifier is a Custom classifier used for the final prediction
 
===============================================================================
"""
class StudentizedOLSClassifier(BaseEstimator, ClassifierMixin ):
    
    def __init__(self, threshold = 'threshold'):
        
        #decision threshold of studentized residuals
        self.threshold = threshold    
        np.random.seed(SEED)
        
    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y) 
        
        #convert to df
        self.X_ = pd.DataFrame(X)
        self.y_ = pd.DataFrame(y)         
        
        #Fit OLS model
        self.ols_mod = OLS(endog = self.y_, exog = self.X_)
        self.ols_result = self.ols_mod.fit()
        
        # Return the classifier
        return self

    def predict(self, X):
        
        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X)
        X_n = pd.DataFrame(X)
        
        #OLS prediction       
        prediction = self.ols_result.predict(X_n)        
        
        #calculate outlier and influence measures for OLS result
        inf = OLSInfluence(self.ols_result)
        
        #Staandard Deviation of studentized residuals
        std = inf.resid_std
        
        """
        Subtract the median of the predictions from the predictions to create 
        an estimated residual. Then divide the estiamted residual by the by 
        the estimated standard deviation, the standard deviation of the 
        residuals from training, to create an estimated studentized residual.
           
        """ 
        # estimated residual
        estimated_residual = prediction - np.nanmedian(prediction)
        
        #estiamted studentized residual
        stud_res = estimated_residual/np.nanmean(std)    #estimate using mean
        #stud_res = prediction/np.nanmedian(std)         #estimate using median
        
        #create predictions based on the threshold
        self.preds = []        
        for res in stud_res:
            if res >= self.threshold:    
                self.preds.append(True)
            else:
                self.preds.append(False)
                
        return self.preds

"""
===============================================================================
                        Main
 
===============================================================================
"""

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
       'total_payments', 'exercised_stock_options', 'bonus',
       'restricted_stock', 'shared_receipt_with_poi',
       'restricted_stock_deferred', 'total_stock_value', 'expenses',
       'loan_advances', 'from_messages', 'other',
       'from_this_person_to_poi', 'director_fees', 'deferred_income',
       'long_term_incentive', 'from_poi_to_this_person'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)  
 
# Create Pandas DataFrame
df = pd.DataFrame.from_dict(data_dict, orient = 'index')


"""
===============================================================================
                        Remove outliers
 
===============================================================================
"""
### Task 2: Remove outliers

#Drop email column, I will not be using it
df.drop(['email_address'], axis = 1, inplace = True)

# convert all columns to numeric
cols = df.columns
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# Fill Nans with 0, all other methods caused data leakage
df = df.fillna(value = 0, axis = 0)

# split X and y
y = df['poi'].copy()
df.drop(['poi'], axis = 1, inplace = True)

# Find Outliers using Ordinary Least Squares
ols_mod = OLS(endog = y.values, exog = df)
ols_result = ols_mod.fit()

inf = OLSInfluence(ols_result)
stud = inf.resid_studentized_internal

less_outliers = list(stud[stud < -1 ].index) 
more_outliers = list(stud[stud > 1 ].index)

# Add less outliers to drop list
drop_list = set(df.loc[less_outliers].index )

# Add total 'THE TRAVEL AGENCY IN THE PARK' as described in official docs and
# KAMINSKI WINCENTY J, who was in the more outliers but not a poi
drop_list.update(df.loc[['TOTAL']].index)
drop_list.update(df.loc[['THE TRAVEL AGENCY IN THE PARK']].index)
drop_list.update(df.loc[['KAMINSKI WINCENTY J']].index)

# Drop rows
df.drop(drop_list, axis = 0, inplace = True)
y.drop(drop_list, axis = 0, inplace = True)

# Combine into one df with poi first
df_all = pd.concat([y, df ], axis = 1)

data_dict = df_all.to_dict('index')    
my_dataset = data_dict
"""
===============================================================================
                        Create new feature(s)
 
===============================================================================
"""
### Task 3: Create new feature(s)
# no new features used in the final version refer to 
# Enron_Notes_Final_Project.html for all features

"""
===============================================================================
                        Final Model
 
===============================================================================
"""
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#ols_m = StudentizedOLSClassifier(1.04)
clf = Pipeline([('kBest', SelectKBest( k = 19)), ('ols', StudentizedOLSClassifier(1.04))])

"""
===============================================================================
                        tester.py for local use
 
    Uncomment to run local test
===============================================================================
"""

# =============================================================================
# def tester_prep(dfn):
#     features_list = dfn.columns.values
#     data_dict = dfn.to_dict('index')
#     return features_list, data_dict
# 
# feat, dat = tester_prep(df_all)
# test_classifier(clf, dat, feat)
# =============================================================================

"""
===============================================================================
                        pkl's
 
===============================================================================
"""
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Tuning is in  Enron_Notes_Final_Project.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)