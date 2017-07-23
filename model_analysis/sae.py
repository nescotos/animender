import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from types import *
import numpy as np
import sys
from time import time

class SAE(nn.Module):
    def __init__(self, input_output_size, encoder_input=20, decoder_input=20):
        super(SAE, self).__init__()
        self.encoder = nn.Linear(input_output_size, encoder_input)
        self.hidden_layers = []
        self.decoder = nn.Linear(decoder_input, input_output_size)
        self.activation = nn.Sigmoid()
        self.last_out = encoder_input
        
    def forward(self, x):
        #First step
        x = self.activation(self.encoder(x))
        #We only handle Linear and Dropout layers
        for layer in self.hidden_layers:
            if "Linear" in str(type(layer)):
                #It's a linear layer
                x = self.activation(layer(x))
            else:
                #It's a dropout layer
                x = layer(x)
        #Final Step
        x = self.decoder(x)
        return x
    
    def add_hiden_layer(self, out_features):
        new_layer = nn.Linear(self.last_out, out_features)
        self.last_out = out_features
        self.hidden_layers.append(new_layer)
        
    def add_dropout(self, p=0.5):
        new_dropout = nn.Dropout(p)
        self.hidden_layers.append(new_dropout)
        
    def print_progress(self, message):
        sys.stdout.write("\r" + message)
        sys.stdout.flush()
    def compile(self, optimizer='rmsprop', criterion='mse', lr=0.01, weight_decay=0.5):
        if type(criterion) is StringType:
            if criterion == 'mse':
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.L1Loss()
        else:
            self.criterion = criterion
        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)           
            
            
    def fit(self, X, nb_epoch):
        init_time = time()
        for epoch in range(nb_epoch):
            train_loss = 0
            s = 0.
            rows, columns = X.size()
            init_iteration_time = time()
            init_training_time = time()
            for index in range(int(rows)):
                input = Variable(X[index]).unsqueeze(0)
                target = input.clone()
                if torch.sum(target.data > 0) > 0:
                    output = self.forward(input)
                    target.require_grad = False
                    output[target == 0] = 0
                    loss = self.criterion(output, target)
                    mean_corrector = columns/float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.data[0] * mean_corrector)
                    s += 1.
                    self.optimizer.step()
                end_training_time = time()
                self.print_progress("epoch: {0}/{1}, training: {2}/{3} - {4:.2%}, time: {5:.2f}s".format(epoch + 1, nb_epoch,index + 1, rows, (index + 1)/float(rows), end_training_time - init_training_time))
            end_iteration_time = time()
            self.print_progress('epoch: {0}/{1}, training loss: {2}, total epoch time: {3:.2f}s\n'.format(epoch + 1, nb_epoch, train_loss / s, end_iteration_time - init_iteration_time))
        end_time = time()
        self.print_progress("Total Training Time: {0:.2f}m".format((end_time - init_time) / float(60)))
            
    def perform(self, X, y):
        test_loss = 0
        s = 0.
        rows, columns = X.size()
        for index in range(int(rows)):
            input = Variable(X[index]).unsqueeze(0)
            target = Variable(y[index]).unsqueeze(0)
            if torch.sum(target.data > 0) > 0:
                output = self.forward(input)
                target.required_grad = False
                output[target == 0] = 0
                loss = self.criterion(output, target)
                mean_corrector = columns/float(torch.sum(target.data > 0) + 1e-10)
                test_loss += np.sqrt(loss.data[0] * mean_corrector)
                s += 1.
        print "prediction loss {0}".format(test_loss/s) 

    def predict(self, X):
        prediction = []
        rows, _ = X.size()
        for index in range(int(rows)):
            input = Variable(X[index]).unsqueeze(0)
            output = self.forward(input)
            prediction.append(output)
        return prediction
        
   