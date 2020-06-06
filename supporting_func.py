import numpy as np
import random

def R_logistic(y):
    y1 = 1 - y
    R = np.multiply(y,y1)
    R = np.matrix(R)
    return np.diag(R.A1)

def sigmoid(a):
    y = 1 / (1 + np.exp(-a))
    return y


def calculateYR(a, algo):
    if(algo == "log"):
        y = sigmoid(a)
        R = R_logistic(y)
    return y, R

def predict(Wmap, test_x, algo):
    a = np.matmul(test_x, Wmap)
    if(algo == "log"):
        t = sigmoid(a)
        t_hat = [1 if k>=0.5 else 0 for k in t]
        return t_hat

def calculate_err(t, t_hat, algo):
    if(algo == "log"):
        err = []
        for i in range(len(t)):
            if(t[i] == t_hat[i]):
                err.append(0)
            else:
                err.append(1)
        final_err = np.mean(err)
        return final_err
        
def calculate_err2(t, t_hat, algo):
    if(algo == "log"):
        count = 0
        for i in range(len(t)):
            if(t[i] == t_hat[i]):
                count += 1
        final_err = count / len(t_hat)
        return final_err

def GetRandomSample(train_x,train_y, nf):
    samp_x = []
    samp_y = []    
    sample_size = int(len(train_y) * nf)    
    temp_array = list(range(0,len(train_x)))
    c = random.sample(temp_array,sample_size)
    for index in c:
        samp_x.append(train_x[index])
        samp_y.append(train_y[index])
    samp_x = np.stack(samp_x, axis=0)
    samp_y = np.matrix.transpose(np.stack(samp_y, axis=1))
    return samp_x,samp_y,sample_size

def divideData(n_data, labels):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    sample_size = int(len(labels)/3)
    temp_array = list(range(0, len(labels)))
    c = random.sample(temp_array,sample_size)
    for i in range(len(labels)):
        if(i in c):
            test_x.append(n_data[i])
            test_y.append(labels[i])
        else:
            train_x.append(n_data[i])
            train_y.append(labels[i])
    return np.asmatrix(train_x), np.asmatrix(train_y), np.asmatrix(test_x), np.asmatrix(test_y)

def calculate_statistics(no_of_iterations, total_err):
    final_iters = []
    mean_err = []
    std = []
    for i in range(0,10):
        collector  = []
        collector2 = []
        for j in range(0,30):
            collector.append(no_of_iterations[j][i])
            collector2.append(total_err[j][i])
        it = np.mean(collector)
        mu = np.mean(collector2)
        sd = np.std(collector2)
        final_iters.append(it)
        mean_err.append(mu)
        std.append(sd)
    return mean_err, std, final_iters