import os
import random

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import *
import h5py

from model import *
import config

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from sklearn.metrics import recall_score

class Miir(data.Dataset):
    def __init__(self, data_path=config.DATA_PATH, train=True):
        self.train = train
        #print(data_path)
        self.train_features = np.load(data_path[0])
        self.train_label = np.load(data_path[1])
        self.test_features = np.load(data_path[2])
        self.test_label = np.load(data_path[3])
        self.train_size = len(self.train_features)
        self.test_size = len(self.test_features)
        self.train_features = self.train_features.reshape(self.train_features.shape[0],self.train_features.shape[1],self.train_features.shape[2],1)
        #print(self.train_features.shape)
        self.test_features = self.test_features.reshape(self.test_features.shape[0],self.test_features.shape[1],self.test_features.shape[2],1)
        self.train_features = list(self.train_features)
        self.test_features = list(self.test_features)

        #print(self.train_features[0].shape)
    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size

    def __getitem__(self, idx):
        if self.train:
            return self.train_features[idx], self.train_label[idx]
        else:
            return self.test_features[idx], self.test_label[idx]



def loss_fn(target, pred_target, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    """
    :param target:
    :param pred_target
    :param f_mean:
    :param f_logvar:
    :param z_post_mean:
    :param z_post_logvar:
    :param z_prior_mean:
    :param z_prior_logvar:
    :return:
    Loss function consists of 3 parts, Cross Entropy of the predicted targes and the target, the KL divergence of f,
    and the sum over the KL divergence of each z_t, with the sum divided by batch_size.
    Loss = {CrossEntropy(pred_target, target) + KL of f + sum(KL of z_t)}/batch_size
    Prior of f is a spherical zero_mean unit variance Gaussian and the prior for each z_t is a Gaussian whose
    mean and variance are given by LSTM.
    """
    batch_size = target.size(0)
    #print(target,pred_target)

    pred_target = torch.reshape(pred_target,(pred_target.shape[0],))
    #print(pred_target) 
    #print(pred_target)

    cross_entropy = F.binary_cross_entropy_with_logits(torch.abs(pred_target), target)
    
    #raise
    #print(cross_entropy)
    kld_f = 1*torch.sum(1+f_logvar - torch.pow(f_mean,2) -0.2*torch.exp(f_logvar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z =  10*torch.mean(z_prior_logvar - z_post_logvar -5*((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2))
                                                               / z_prior_var) - 1)

    return (cross_entropy + (kld_f + kld_z)) / batch_size, kld_f / batch_size, kld_z / batch_size,\
           cross_entropy/batch_size


def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'opimizer': optim.state_dict()}, path)


def check_accuracy(model, test):
    model.eval()
    total = 0
    count=0
    correct_target = 0
    with torch.no_grad():
        for item in test:
            features, target = item
            batch_size = target.size(0)
            #target = torch.argmax(target, dim=1) # one-hot back to int
            *_, pred_target = model(features)
            pred_target = torch.reshape(pred_target,(batch_size,))
            print(pred_target[50:100])
            print(pred_target.std())
            print(target[50:100])
            print(torch.mean(pred_target))
            for i in range(len(pred_target)):

                #if pred_target[i]>torch.mean(pred_target):
                if pred_target[i]>torch.sum(torch.mean(pred_target)-pred_target.std()):
                    pred_target[i] = 1
                else:
                    pred_target[i] = 0
        
            #print(target)

            count+=1
            recall = recall_score(target,pred_target,average='macro')
            #print(recall)
            #raise

    model.train()
    return recall


def train_classifier(model, optim, dataset, epochs, path, test, start = 0):
    model.train()
    for epoch in range(start, epochs):
        losses = []
        kld_fs = []
        kld_zs = []
        cross_entropies = []
        
        for item in tqdm(dataset):
            features, target = item
            # one hot back to int
            optim.zero_grad()
            #print(features.shape)
            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean,\
            z_prior_logvar, pred_target = model(features)
            loss, kld_f, kld_z, cross_entropy = loss_fn(target, pred_target, f_mean, f_logvar,
                                                       z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            kld_fs.append(kld_f.item())
            kld_zs.append(kld_z.item())
            cross_entropies.append(cross_entropy.item())

        test_accuracy = check_accuracy(model, test)
        meanloss = np.mean(losses)
        meanf = np.mean(kld_fs)
        meanz = np.mean(kld_zs)
        mean_cross_entropies = np.mean(cross_entropies)
         #print out result every 20 epochs
        print("Epoch {} : Average Loss: {} KL of f : {} KL of z : {} "
                  "Cross Entropy: {} Test Accuracy: {}".format(epoch + 1, meanloss, meanf, meanz, mean_cross_entropies,
                                                               test_accuracy))
        save_model(model, optim, epoch, path)


if __name__=='__main__':
    model = DisentangledEEG(factorized=False, nonlinearity=True)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_data = Miir(config.DATA_PATH, True)
    test_data = Miir(config.DATA_PATH, False)
    loader = data.DataLoader(train_data, batch_size=64, num_workers=1)
    loader_test = data.DataLoader(test_data, batch_size=11398, shuffle=True, num_workers=4)
    train_classifier(model=model, optim=optim, dataset=loader, epochs=200,
                     path='./checkpoint_disentangled_alco.pth',test = loader_test)