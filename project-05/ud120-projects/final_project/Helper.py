import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import mean
import pandas as pd

# Get feature list
def get_feature_list():
    return [
        'poi',
        'salary',
        'bonus',
        'deferral_payments',
        'deferred_income',
        'director_fees',
        'exercised_stock_options',
        'expenses',
        'loan_advances',
        'long_term_incentive',
        'other',
        'restricted_stock',
        'restricted_stock_deferred',
        'salary',
        'total_payments',
        'total_stock_value',
        'from_messages',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
        'shared_receipt_with_poi',
        'to_messages'
    ]
# Get poi count based on the passed in data_dict
def get_poi_count(data_dict):
    poi = 0
    for name in data_dict.keys():
        if data_dict[name]['poi'] == True:
            poi += 1
    return poi

#Print data set details like number of poi,total data poins, etc
def print_data_set_details(data_dict,poi,all_features):
    print('-------------------')
    print('Dataset details')
    print('-------------------')
    print('Total data points %d' % len(data_dict.keys()))
    print('Number of POI: %d' % poi)
    print('Number of non POI: %d' % (len(data_dict.keys()) - poi))
    print('Total number of features  %d' %  len(all_features))

#Get missing values count
def get_missing_values_count(data_dict, all_features):
    missing_values = {}
    for feature in all_features:
        missing_values[feature] = 0
    for person in data_dict.keys():
        for feature in all_features:
            if data_dict[person][feature] == 'NaN':
                missing_values[feature] += 1
    return missing_values

# print missing values features count
def print_missing_values_features_count(all_features,missing_values):
    print('-------------------')
    print('Missing values of all the features')
    print('-------------------')
    for feature in all_features:
        print("%s: %d" % (feature, missing_values[feature]))

# print missing values features count
def PlotOutlier(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        current_x = point[0]
        current_y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(current_x, current_y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

#Remove outliers
def remove_outlier(dict_object, keys):
    for key in keys:
        dict_object.pop(key, 0)
    return dict_object

#Compute fraction
def compute_fraction(poi_messages, all_messages):
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.
    fraction = poi_messages / all_messages
    return fraction

#Add new features
# def add_new_features(my_dataset):
#     for name in my_dataset:
#         data_point = my_dataset[name]
#         from_poi_to_this_person = data_point["from_poi_to_this_person"]
#         to_messages = data_point["to_messages"]
#         fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
#         data_point["fraction_from_poi"] = fraction_from_poi
#         from_this_person_to_poi = data_point["from_this_person_to_poi"]
#         from_messages = data_point["from_messages"]
#         fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
#         data_point["fraction_to_poi"] = fraction_to_poi
#     return my_dataset

#Get new feature list
def get_new_feature_list():
    return ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                                     'shared_receipt_with_poi', 'fraction_to_poi']
#Get k best
def get_k_best(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features)
    return k_best_features

#Evaluate clf  and pring
def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing Please wait..')
            sys.stdout.flush()
            first = False

    print('\n-------------------')
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    print('-------------------')


    return mean(precision), mean(recall)

# Dump data to a file based on the path
def dump_data_to_file(data,path):
    pickle.dump(data, open(path, "w"))

# Dump all data to files
def dump_all_data_to_files(clf,my_dataset,my_feature_list):
    dump_data_to_file(clf, "my_classifier.pkl")
    dump_data_to_file(my_dataset, "my_dataset.pkl")
    dump_data_to_file(my_feature_list, "my_feature_list.pkl")
