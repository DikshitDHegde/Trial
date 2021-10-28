import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils.linear_assignment_ import linear_assignment
from tqdm import tqdm
from torch.utils.data import Dataset


def save_latent(dir, epoch, latent, y_k, name):
    data = {
        "latent": latent,
        "kmeans": y_k}
    with open(os.path.join(dir, f"{name}_{epoch}_.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def set_seed_globally(seed_value=0, if_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if if_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = True

def load_data(datapath):
    f = np.load(datapath)
    x,y = f['x'], f['y']
    return x,y

class dataSet(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.x, self.y = load_data(datapath)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.x[index])) , torch.from_numpy(np.array(self.y[index])), torch.from_numpy(np.array(index))

class optimizer():
    def __init__(self, model, lr, optimizer='adam'):
        params = model.parameters()
        if optimizer.lower() == 'adam':
            self.optim = optim.Adam(params=params, lr=lr,betas=(0.9,0.999))
        if optimizer.lower() == 'sgd':
            self.optim = optim.SGD(params=params, lr=lr)


    def call(self):
        return self.optim


def test(model, dataloader , device="cpu",save_latents=None, epoch=10):
    model.eval()
    features = []
    label = []
    for i, data in tqdm(enumerate(dataloader, 0),leave=False):
        inputs, labels,_ = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.reshape(inputs.size(0), -1)
        z, _ = model(inputs)
        features.extend(z.detach().cpu().numpy())
        label.extend(labels.detach().cpu().numpy())

    kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=-1, random_state=42)
    y_pred = kmeans.fit_predict(features)
    label = np.array(label)
    nmi = nmi_score(y_pred, label)
    acc = cluster_acc(y_pred, label)
    ari = ari_score(y_pred, label)
    if save_latents != None:
        save_latent(save_latents, epoch, features, y_pred, "Z")
    return acc, nmi, ari