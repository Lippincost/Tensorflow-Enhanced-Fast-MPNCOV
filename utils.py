import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def plot_curve(stats, path, iserr):
    """plot curve of loss and accuracy"""
    train_loss = np.array(stats['train']['loss'])
    test_loss = np.array(stats['val']['loss'])
    if iserr:
        trainTop1 = 100 - np.array(stats['train']['top1'])
        trainTop5 = 100 - np.array(stats['train']['top5'])
        valTop1 = 100 - np.array(stats['val']['top1'])
        valTop5 = 100 - np.array(stats['val']['top5'])
        titleName = 'error'
    else:
        trainTop1 = np.array(stats['train']['top1'])
        trainTop5 = np.array(stats['train']['top5'])
        valTop1 = np.array(stats['val']['top1'])
        valTop5 = np.array(stats['val']['top5'])
        titleName = 'accuracy'
    epoch = len(train_loss)
    figure = plt.figure()
    obj = plt.subplot(1, 3, 1)
    obj.plot(range(1, epoch + 1), train_loss, 'o-', label='train')
    obj.plot(range(1, epoch + 1), test_loss, 'o-', label='val')
    plt.xlabel('epoch')
    plt.title('objective')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(1, 3, 2)
    top1.pl