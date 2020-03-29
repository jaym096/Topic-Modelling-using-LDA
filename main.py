import sys
import random
import csv
import numpy as np
import timeit
import glm
import copy
import matplotlib.pyplot as plt
import supporting_func as func

err_arr = []

def init(path, dataset_len, k):
    unique_vocab = []
    all_vocab = []
    word_occur = {}
    d_n = []
    z_n = []
    w_n = []
    for fn in range(1, dataset_len+1):
        w = []
        with open(path+str(fn), "r") as file:
            for line in file.read().split("\n"):
                for word in line.strip().split():
                    w.append(word)
                    d_n.append(fn-1)
                    z_n.append(random.randint(0,k-1))
                    if word in unique_vocab:
                        all_vocab.append(word)
                    else:
                        all_vocab.append(word)
                        unique_vocab.append(word)
                    w_n.append(unique_vocab.index(word))
        word_occur.update({fn:w})
    return unique_vocab, all_vocab, d_n, z_n, w_n, word_occur
            
def createBOW(unique_vocab, word_occur):
    b_o_w = np.zeros((len(word_occur), len(unique_vocab)))
    for i in range(len(word_occur)):
        docwords = word_occur[i+1]
        #print(docwords)
        for j in range(len(unique_vocab)):
            if(unique_vocab[j] in docwords):
                b_o_w[i][j] = docwords.count(unique_vocab[j])
    return b_o_w

def task1(dataset, path):
    n_iters = 500
    beta = 0.01
    k = 2
    alpha = 5/k
    dataset_len = 0
    if(dataset=="artificial"):
        k = 2
        alpha = 5/k
        dataset_len = 10
    if(dataset=="20newsgroups"):
        k = 20
        alpha = 5/k
        dataset_len = 200
    
    #initialize required arrays
    unique_vocab, full_vocab, d_n, z_n, w_n, word_occur = init(path, dataset_len, k)
    
    #generate a random permutation of pi_n
    pi_n = list(np.arange(0, len(full_vocab), 1))
    random.shuffle(pi_n)
    
    #initialize c_d & c_t
    c_d = np.zeros((dataset_len, k))
    c_t = np.zeros((k, len(unique_vocab)))
    for i in range(len(full_vocab)):
        c_d[d_n[i]][z_n[i]] += 1
        c_t[z_n[i]][w_n[i]] += 1
    
    #initialize array of probabilities
    p = [0] * k

    #Collapsed Gibbs Sampler for LDA
    for i in range(n_iters):
        for n in range(len(full_vocab)):
            word = w_n[pi_n[n]]
            topic = z_n[pi_n[n]]
            doc = d_n[pi_n[n]]
            c_d[doc][topic] = c_d[doc][topic]-1
            c_t[topic][word] = c_t[topic][word]-1
            for k_ in range(k):
                num = (c_t[k_][word] + beta) * (c_d[doc][k_] + alpha)
                #denom = (len(unique_vocab) * beta) + (np.sum(c_t[k_,:])*k*alpha) + (np.sum(c_d[doc,:]))
                denom = ((len(unique_vocab) * beta) + (np.sum(c_t[k_,:]))) * ((k*alpha) + (np.sum(c_d[doc,:])))
                p[k_] = num/denom
            p = np.divide(p, np.sum(p))
            topic = np.random.choice(range(0,k),p=p)
            z_n[pi_n[n]] = topic
            c_d[doc][topic] += 1
            c_t[topic][word] += 1
    
    #getting the top words
    topics = create_dict(c_t, unique_vocab, dataset)
    print("tops words are written into the csv file!!")
    with open("topicwords.csv", "w", newline="") as file:
        for t in topics:
            writer = csv.writer(file)
            writer.writerow(topics[t])
    file.close()
            
    #Calculating topic representation
    t_rep = copy.deepcopy(c_d)
    for i in range(c_d.shape[0]):
        for j in range(k):
            #a = 0.01 #alpha
            denom = (k * alpha) + np.sum(t_rep[i,:])
            num = t_rep[i][j] + alpha
            t_rep[i][j] = num / denom
            
    return z_n, c_d, c_t, unique_vocab, t_rep, word_occur

def create_dict(c_t, unique_vocab, dataset):
    topics={}
    if(dataset == "artificial"):
        for i in range(len(c_t)):
            for j in c_t[i].argsort()[-3:][::-1]:
                if i in topics:
                    topics[i]+= [unique_vocab[j]]
                else:
                    topics[i] = [unique_vocab[j]]
    if(dataset == "20newsgroups"):
        for i in range(len(c_t)):
            for j in c_t[i].argsort()[-5:][::-1]:
                if i in topics:
                    topics[i]+= [unique_vocab[j]]
                else:
                    topics[i] = [unique_vocab[j]]
    return topics

def logistic(n_data, labels, algo, alpha):
    total_err = []
    no_of_iterations = []
    n = np.arange(0.1,1.1,0.1)
    for i in range(0,30):
        train_x, train_y, test_x, test_y = func.divideData(n_data, labels)
        size_err = []
        sizes = []
        iterations_for_sizes = []
        for i in range(len(n)):
            samp_x, samp_y, sample_size = func.GetRandomSample(train_x, train_y, n[i])
            sizes.append(sample_size)
            shp = np.shape(samp_x)
            w = np.zeros((shp[1],1))
            Wmap, iterations = glm.GLM(samp_x, samp_y, w, alpha, algo)
            t_hat = func.predict(Wmap, test_x, algo)
            err = func.calculate_err2(test_y, t_hat, algo)
            size_err.append(err)
            iterations_for_sizes.append(iterations)
        total_err.append(size_err)
        no_of_iterations.append(iterations_for_sizes)
    
    mean_err, sd_err, avg_iterations = func.calculate_statistics(no_of_iterations, total_err)
    
    return mean_err, sd_err, sizes
    

if __name__ == "__main__":
    dataset = "20newsgroups"
    path = dataset+"/"
    print("Please wait the code is running...")
    t1 = timeit.default_timer()
    z_n, c_d, c_t, unique_vocab, t_rep, word_occur = task1(dataset, path)
    t2 = timeit.default_timer()
    print("Time taken to run task1 : ", t2-t1)
    
    #creating bag of words
    bag_of_words = createBOW(unique_vocab, word_occur)
    
    #get labels
    temp_lab = np.genfromtxt(path+"index.csv", delimiter=',',dtype=int)[:,[1]]
    labels = np.array(temp_lab)
    
    t1 = timeit.default_timer()
    #Logistic regression on bags of words
    mean_err1, sd_err1, sizes = logistic(bag_of_words, labels, "log", alpha=0.01)
    
    #Logistic regression on LDA
    mean_err2, sd_err2, sizes = logistic(t_rep, labels, "log", alpha=0.01)
    t2 = timeit.default_timer()
    print("Time taken to run task2: ", t2-t1)
    
    #plotting
    plt.errorbar(sizes, mean_err2, sd_err2, linewidth=1.0, label="LDA")
    plt.errorbar(sizes, mean_err1, sd_err1, linewidth=1.0, label="Bag of Words")
    plt.xticks(sizes, [".1N",".2N",".3N",".4N",".5N",".6N",".7N",".8N",".9N","1N"])
    plt.xlabel('size of training data set')
    plt.ylabel('Performance')
    plt.title("Learning curve for LDA vs Bag of Words as function of increasing training set",
              fontsize="small")
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()
    
    

    