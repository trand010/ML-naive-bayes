#Daniel Tran
#Machine Learning Program#2

import numpy as np
from sklearn import linear_model
import os

def load_data(filename='spambase.data'):
    #load the data from file 'spambase.data'
    data = np.loadtxt(filename, delimiter=',', unpack=False)
    data_x, data_y = data.shape
    #split the data to have equal positive and negative parts
    split = 0.0
    while split <= 0.98 or split >= 1.02:
        np.random.shuffle(data)
        #cut data in half for training and testing sets
        test_data = data[:int(data_x/2)]
        train_data = data[int(data_x/2):]
        #get the spam split of each set and if balanced exit out
        n_spam_r = train_data[:,-1].sum()
        n_spam_t = test_data[:,-1].sum()
        split = 2*n_spam_r / (n_spam_r + n_spam_t)
    #change 0 to -1 and take truth column out
    rt = train_data[:,-1]
    train_data = np.delete(train_data, -1, 1)
    tt = test_data[:,-1]
    test_data = np.delete(test_data, -1, 1)
    return train_data, rt, test_data, tt

def conditional_probability(train_data, rt):
    data_x, data_y = train_data.shape
    #split postive and negative samples
    pos_train = train_data[rt==1]
    neg_train = train_data[rt==0]
    #get mean of postive and negative
    m_train = np.empty((2,data_y))
    m_train[0,:] = np.mean(neg_train, axis=0)
    m_train[1,:] = np.mean(pos_train, axis=0)
    #standard deviation of positive and negative
    strain_data = np.empty((2,data_y))
    strain_data[0,:] = np.std(neg_train, axis=0)
    strain_data[1,:] = np.std(pos_train, axis=0)
    return m_train, strain_data 

def naive_bayes(test_data, m, sd, ppos, pneg):
    #remove columns with 0 std
    b = np.where(sd[1,:] == 0) 
    m = np.delete(m_train, b, 1)
    sd = np.delete(strain_data, b, 1)
    test_data = np.delete(test_x, b, 1)
    #get constant values
    data_x, data_y = test_data.shape
    lr2pi = np.log(2*np.pi)/(-2)
    lpn = np.log(pneg)*(-2) - np.log(2*np.pi)
    lps = np.log(ppos)*(-2) - np.log(2*np.pi)
    var = np.power(sd, 2)
    #get prediction for each instance
    prediction = np.empty(data_x)
    for i in range(data_x):
        #get con prob for negative cases
        a_lpn = np.power((test_data[i,:]-m[0,:]),2) / var[0,:]
        a_lpn += np.log(var[0,:])
        #add prior prob to sum
        neg = lpn + np.sum(a_lpn)
        #get con prob for positive
        alps = np.power((test_data[i,:]-m[1,:]),2) / var[1,:]
        alps += np.log(var[1,:])
        #add prior probability to sum
        pos = lps + np.sum(alps)
        #return the smaller value
        if pos <= neg:
            prediction[i] = 1
        else:
            prediction[i] = 0
    return prediction

def check_prediction(predict, actual):
    #make confusion matrix get accuracy, percision, and recall
    cmatrix = np.zeros([2, 2])
    for i in range(len(predict)):
        if predict[i] > actual[i]:  
            cmatrix[1, 0] += 1
        elif predict[i] < actual[i]:
            cmatrix[0, 1] += 1
        elif predict[i] == 1:       
            cmatrix[0, 0] += 1
        else:
            cmatrix[1, 1] += 1
    accuracy = np.trace(cmatrix) / np.sum(cmatrix)
    precision = np.sum(cmatrix[0, 0]) / np.sum(cmatrix[:,0])
    recall = np.sum(cmatrix[0, 0]) / np.sum(cmatrix[0,:])
    return accuracy, precision, recall, cmatrix 

#split into training and testing 
train_x, train_t, test_x, test_t = load_data()
#get conditional probability
m_train, strain_data = conditional_probability(train_x, train_t)
#spam probability
ppos = np.sum(train_t) / len(train_t)
#non-spam probability
pneg = 1 - ppos
#naive bayes prediction
prediction = naive_bayes(test_x, m_train, strain_data, ppos, pneg)
#get accuracy, precision, and recall
accu, prec, recall, cmatrix = check_prediction(prediction, test_t)
#clear the screen and display results
os.system('cls' if os.name == 'nt' else 'clear')
print(f'Accuracy = {accu}')
print(f'Precision = {prec}')
print(f'Recall = {recall}')
print(f'Confusion Matrix:\n {cmatrix} \n\n')