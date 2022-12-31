import torch
import itertools
import torch.nn as nn
import numpy as np
import os
import pathlib
from os.path import exists
import matplotlib.pyplot as plt
from models import *

classes_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def get_checkpoint_path(model_str, device):
    # 获取 checkpoint 的路径
    checkpoint_folder = str(os.path.join(pathlib.Path(__file__).parent.resolve(),
                    'checkpoints', model_str, str(device)))
    os.makedirs(checkpoint_folder, exist_ok=True)
    return str(os.path.join(checkpoint_folder, 'best_checkpoint.pth'))

def select_device(device=''):
    # 选择设备
    if device.lower() == 'cuda':
        if not torch.cuda.is_available():
            print ("torch.cuda not available")
            return torch.device('cpu')    
        else:
            return torch.device('cuda:0')
    if device.lower() == 'dml':
        return torch.device('dml')
    else:
        return torch.device('cpu')

def get_model(model_str, device, load_checkpoint=False):
    # 获取模型
    if (model_str == 'resnet18'):
        model = ResNet18().to(device)
    elif (model_str == 'vgg19'):
        model = VGG('VGG19').to(device)
    else:
        raise Exception(f"Model {model_str} is not supported yet!")

    if load_checkpoint:
        checkpoint = get_checkpoint_path(model_str, device)
        if (exists(checkpoint)):
            print(f'Loading checkpoint from: {checkpoint}')
            model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

    return model

def remix(X, y, num_class_list, alpha=1.0, remix_tau=0.5, remix_kappa=1.5):
    
    l = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0))
    X_a, X_b = X, X[idx]
    y_a, y_b = y, y[idx]
    mixed_X = l * X_a + (1 - l) * X_b

    l_list = torch.empty(X.shape[0]).fill_(l).float()
    n_i, n_j = num_class_list[y_a], num_class_list[y_b].float()

    if l < remix_tau:
        l_list[n_i/n_j >= remix_kappa] = 0
    if 1 - l < remix_tau:
        l_list[(n_i*remix_kappa)/n_j <= 1] = 1

    return mixed_X, y_a, y_b, l_list

class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    https://github.com/CoinCheung/pytorch-loss
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()