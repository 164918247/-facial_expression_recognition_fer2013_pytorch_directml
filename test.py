import torch
import dataloader
import utils
import time
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt


def eval(dataloader, model, device, criterion, mode='Val'):
    total = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 切换到评估模式
    model.eval()

    test_loss = 0
    correct = 0

    start = time.time()
    with torch.no_grad():
        for X, y in dataloader:

            # X.shape = [64, 5, 1, 40, 40] 对应 [BatchSize, NCrop, C, H, W]
            batch_size, ncrop, c, h, w = X.shape
            # NCrop之后，一个输入变成N个，需要合一下，转成[BatchSize × NCrop, C, H, W]
            X = X.view(-1, c, h, w)

            X = X.to(device)
            
            # 模型输出
            output = model(X)

            # 损失函数要在CPU算

            # 将NCrop之后的结果求平均值
            output_avg = output.view(batch_size, ncrop, -1).mean(1)
            output_avg = output_avg.to('cpu')
            y = y.to('cpu')

            # 计算损失
            batch_loss = criterion(output_avg, y)
            test_loss += batch_loss.item()

            _, pred = torch.max(output_avg.data, 1)
            correct += pred.eq(y.data).to('cpu').sum().item()

    # 计算总损失
    test_loss = test_loss / num_batches
    accuracy = correct / total

    print(f"{mode} Error: Acc {(100*accuracy):>0.1f}% | Avg loss: {test_loss:>8f} in {time.time() - start:>5f}s")

    del test_loss, batch_loss, output, X

    return accuracy

def test(dataloader, device, model_str):
    # 选择模型
    model = utils.get_model(model_str, device, True)

    # 选择设备，N卡、DML、CPU
    device = utils.select_device(device)

    criterion = utils.LabelSmoothSoftmaxCEV1()

    total = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 切换到评估模式
    model.eval()

    test_loss = 0
    correct = 0

    start = time.time()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            # X.shape = [64, 5, 1, 40, 40] 对应 [BatchSize, NCrop, C, H, W]
            batch_size, ncrop, c, h, w = X.shape
            # NCrop之后，一个输入变成N个，需要合一下，转成[BatchSize × NCrop, C, H, W]
            X = X.view(-1, c, h, w)

            X = X.to(device)
            
            # 模型输出
            output = model(X)

            # 损失函数要在CPU算

            # 将NCrop之后的结果求平均值
            output_avg = output.view(batch_size, ncrop, -1).mean(1)
            output_avg = output_avg.to('cpu')
            y = y.to('cpu')

            # 计算损失
            batch_loss = criterion(output_avg, y)
            test_loss += batch_loss.item()

            _, pred = torch.max(output_avg.data, 1)
            correct += pred.eq(y.data).to('cpu').sum().item()

            # 记录所有分类结果用于绘制混淆矩阵
            if batch == 0:
                all_pred = pred
                all_y = y
            else:
                all_pred = torch.cat((all_pred, pred),0)
                all_y = torch.cat((all_y, y),0)

    # 计算总损失
    test_loss = test_loss / num_batches
    accuracy = correct / total

    # Compute confusion matrix
    matrix = confusion_matrix(all_y.data.to('cpu').numpy(), all_pred.to('cpu').numpy())
    np.set_printoptions(precision=2)

    # 画混淆矩阵
    # Plot normalized confusion matrix

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    plt.figure(figsize=(10, 8))
    utils.plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                        title= f'Test Confusion Matrix (Accuracy: {accuracy*100:0.2f}%)')
    plt.show()

    print(f"Test Error: Acc {(100*accuracy):>0.2f}% | Avg loss: {test_loss:>8f} in {time.time() - start:>5f}s")

def main(path='fer2013.csv', device='dml', model_str='resnet18'):
    test_dataloader = dataloader.get_dataloader(path=path, batch_size=24, input_size=40, mode='Test')
    test(test_dataloader, device, model_str)
    
if __name__ == "__main__":
    main()