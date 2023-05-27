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


# In[7]:


#euclidean distance
train_data_size = len(train_np)
for l in [1,3,7,9,11]:
    print('For l= {},'.format(l))
    counter =  0
    for i,test_data in enumerate(train_np):
        dist = []                                   
        
        new_train_data = np.concatenate((large_np[:i],large_np[i+1:]))
        for train_data in  new_train_data:
            a,b = np.array( [float(i) for i in train_data[:3]]), np.array( [float(i) for i in test_data[:3]])           
             
            euc_d = (np.sum( (a-b)**2 )) **(1/2)  
            dist.append(euc_d)
        
#after finding the euclidean distance and finding the nearest distance using sort      
        near_dist = np.argsort(dist)                
#Finding the nearest k neighbor                                                 
        near_k_nei = [ new_train_data[i][3] for i in near_dist[:l]]
        count = {'W':0,'M':0 }
        for x in near_k_nei:
            if x=='W':
                count['W'] += 1 
            if x=='M':
                count['M'] += 1
        if count['W']>count['M']:
            pred_data ='W'
        else:
            pred_data ='M'
    
 #Finding the accuracy       
        counter +=  1 if pred_data == test_data[3] else 0 
    print('Accuracy : {}'.format(counter/train_data_size))


# In[ ]:




