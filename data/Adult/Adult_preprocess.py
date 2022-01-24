import pandas as pd
from collections import OrderedDict
import numpy as np
import heapq
import random

data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final_weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education_num", "int"),
    ("marital_status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital_gain", "float"),  # required because of NaN values
    ("capital_loss", "int"),
    ("hours_per_week", "int"),
    ("native_country", "category"),
    ("income_class", "category"),
])
target_column = "income_class"

def read_dataset(path):
    return pd.read_csv(
        path,
        names=data_types,
        index_col=None,

        comment='|',  # test dataset has comment in it
        skipinitialspace=True,  # Skip spaces after delimiter
        na_values={
            'capital_gain': 99999,
            'workclass': '?',
            'native_country': '?',
            'occupation': '?',
        },
        dtype=data_types,
    )

def clean_dataset(data):
    # Test dataset has dot at the end, we remove it in order
    # to unify names between training and test datasets.
    data['income_class'] = data.income_class.str.rstrip('.').astype('category')
    
    # Remove final weight column since there is no use
    # for it during the classification.
    data = data.drop('final_weight', axis=1)
    
    # Duplicates might create biases during the analysis and
    # during prediction stage they might give over-optimistic
    # (or pessimistic) results.
    data = data.drop_duplicates()

    return data

def summarize(data):

    attrs = list(data.keys())
    config = {}
    for attr in attrs:
        if str(data[attr].dtype) == 'category' or str(data[attr].dtype) == 'object':
            config[attr] = list(data[attr].unique())
        else:
            config[attr] = [data[attr].unique().min(), data[attr].unique().max()]
    
    return config

def vectorize(data, config):
    # age [17, 90]
    # workclass [State-gov, Self-emp-not-inc, Private, Federal-gov, Local-gov, Self-emp-inc, Without-pay]
    # education ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    #    'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
    #    '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    # education_num [1,16]
    # marital_status [Never-married, Married-civ-spouse, Divorced, Married-spouse-absent,
    #  Separated, Married-AF-spouse, Widowed]
    # occupation, 14 categories
    # relationship
    # race
    # capital_gain
    # capital_loss
    # hours_per_week
    # native country

    # sex
    # income_class

    attris = list(config.keys())
    attris.remove('sex')
    attris.remove('income_class')

    x = []
    label = []
    for i in range(data.shape[0]):
        this_data = data.iloc[i]
        this_x = []
        this_label = []

        if this_data['sex'] == 'Male':
            this_label.append(1)
        else:
            this_label.append(0)
        if this_data['income_class'] == '>50K':
            this_label.append(1)
        else:
            this_label.append(0)
        label.append(np.array(this_label))
            
        for attr in attris:
            if str(data[attr].dtype) == 'category' or str(data[attr].dtype) == 'object':
                idx = config[attr].index(this_data[attr])
                vec = np.zeros(len(config[attr]))
                vec[idx] = 1
                this_x.append(vec)
            else:
                val = np.array((this_data[attr] - config[attr][0]) / (config[attr][1] - config[attr][0])).reshape(1).repeat(14)
                this_x.append(val)
        x.append(np.concatenate(this_x))
    
    x = np.array(x)
    label = np.array(label)
    return x, label

if __name__ == '__main__':

    train_data_file = 'Adult/adult.data'
    test_data_file = 'Adult/adult.test'

    train_data = clean_dataset(read_dataset(train_data_file))
    test_data = clean_dataset(read_dataset(test_data_file))
    
    train_data = train_data[train_data.notnull().all(axis=1)]
    test_data = test_data[test_data.notnull().all(axis=1)]
    data = pd.concat([train_data, test_data])

    config = summarize(data)

    N = data.shape[0]
    random.seed(11)
    idx_adv = random.sample(range(0,N), int(N/3))
    random.seed(22)
    idx_train = random.sample(set(range(0,N)) - set(idx_adv), int(N/3))
    idx_test = list(set(range(0,N)) - set(idx_adv) - set(idx_train))
    adv_data = data.iloc[idx_adv].copy()
    train_data = data.iloc[idx_train].copy()
    test_data = data.iloc[idx_test].copy()

    train_data, train_label = vectorize(train_data, config)
    # adv_data, adv_label = vectorize(adv_data, config)
    test_data, test_label = vectorize(test_data, config)

    # np.savez('Adult_adv.npz',
    #     train_data=adv_data,
    #     train_label=adv_label,
    #     test_data=test_data,
    #     test_label=test_label)
    np.savez('Adult_fair.npz',
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label)
    print('finished!')