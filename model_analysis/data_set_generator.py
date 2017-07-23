import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

def convert_set(data, n_rows, n_cols):
        process_set = []
        for i  in range(n_rows):
            rated = data[data[:, 0] == i]
            animes_rated = rated[:, 1]
            ratings_obtained = rated[:, 2]
            ratings = np.zeros(n_cols)
            ratings[animes_rated] = ratings_obtained
            process_set.append(list(ratings))
            if(i % 4000 == 0):
                print('Iteration {0} of {1}'.format(i, n_rows))
        return process_set 

ratings_corrected = pd.read_csv('./ratings_corrected.csv')
training_set, test_set = train_test_split(ratings_corrected, test_size=0.2, random_state=42)
training_set = training_set.drop(training_set.columns[[0]], axis=1)
test_set = test_set.drop(test_set.columns[[0]], axis=1)
test_set = np.array(test_set, dtype='int')
training_set = np.array(training_set, dtype='int')
print 'Converting Set into a Matrix'
test_set = convert_set(test_set, 68929, 3787)
training_set = convert_set(training_set, 68929, 3787)
print 'Making tensor'
test_set = torch.FloatTensor(test_set)
training_set = torch.FloatTensor(training_set)
print 'Saving tensor'
torch.save(test_set, 'test_set.pkl')
torch.save(training_set, 'training_set.pkl')
print 'Done'

