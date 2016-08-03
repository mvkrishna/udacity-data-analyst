#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import Helper
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = Helper.get_feature_list()

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

all_features = data_dict['ALLEN PHILLIP K'].keys()

poi = Helper.get_poi_count(data_dict)
Helper.print_data_set_details(data_dict, poi, all_features)

missing_values = Helper.get_missing_values_count(data_dict, all_features)
Helper.print_missing_values_features_count(all_features,missing_values)

### Task 2: Remove outliers
print(Helper.PlotOutlier(data_dict, 'total_payments', 'total_stock_value'))
print(Helper.PlotOutlier(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(Helper.PlotOutlier(data_dict, 'salary', 'bonus'))
#Remove outlier TOTAL line in pickle file.
data_dict.pop( 'TOTAL', 0 )
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']

data_dict = Helper.remove_outlier(data_dict, outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for name in my_dataset:
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = Helper.compute_fraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = Helper.compute_fraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

my_feature_list = features_list+ Helper.get_new_feature_list();
num_features = 10
best_features = Helper.get_k_best(my_dataset, my_feature_list, num_features)

my_feature_list = ['poi'] + best_features.keys()

print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])
data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

##########################Task 4: Using algorithm########################


l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

k_clf = KMeans(n_clusters=2, tol=0.001)

s_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'auto')

rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

Helper.evaluate_clf(l_clf, features, labels)
Helper.evaluate_clf(k_clf, features, labels)
Helper.evaluate_clf(s_clf, features, labels)
Helper.evaluate_clf(rf_clf, features, labels)

### Select Logistic Regression as final algorithm
clf = l_clf


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)

# anyone can run/check your results
Helper.dump_all_data_to_files(clf,my_dataset,my_feature_list)
