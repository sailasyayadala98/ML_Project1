#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np

training_data = './dataset/1a-training.txt'
test_data = './dataset/1a-test.txt'
large_120_data = './dataset/1c-data.txt'

train_np = readFile(training_data)
print(train_np)
test_np = readFile(test_data)
print(test_np)
large_np = readFile(large_120_data)


# In[5]:


#calculating the Euclidean distance for given data set
for l in [1,3,7]:
    print('For l= {}, metric = Euclidean'.format(l))
    for test_data in test_np:
        dist = []      
        for train_data in  train_np:
            a,b = np.array( [float(j) for j in train_data[:3]]) , np.array( [float(j) for j in test_data[:3]])
            euc_d = (np.sum( (a-b)**2 ))**(1/2)   
            dist.append(euc_d)
        
#after finding the euclidean distance and finding the nearest distance using sort   
        near_dist = np.argsort(dist)                
#Finding the nearest k neighbor                                                 
        near_k_nei = [ train_np[i][3] for i in near_dist[:l]]
        count = {'W':0,'M':0 }
        for x in near_k_nei:
            if x=='W':
                count['W'] += 1 
            if x=='M':
                count['M'] += 1
        if count['W']>count['M']:
            print('W')
        else:
            print('M')
        


# In[ ]:




