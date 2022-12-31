import torch
import time
import utils
from models import *

def train(dataloader, model, device, criterion, optimizer, accumulation_steps):
    # 切换到训练模式
    model.train()

    train_loss = 0
    correct_a = 0
    correct_b = 0
    correct = 0
    total = 0

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    num_class_list = dataloader.dataset.num_class_list
    num_class_list = torch.FloatTensor(num_class_list)
    
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):

        # X.shape = [64, 5, 1, 40, 40] 对应 [BatchSize, NCrop, C, H, W]
        batch_size, ncrop, c, h, w = X.shape
        # NCrop之后，一个输入变成N个，需要合一下，转成[BatchSize × NCrop, C, H, W]
        X = X.view(-1, c, h, w)
        y = torch.repeat_interleave(y, repeats=ncrop, dim=0)

        X = X.to(device)

        # remix
        mixed_X, y_a, y_b, l_list = utils.remix(X, y, num_class_list)

        # 模型输出
        output = model(mixed_X)
        output = output.to('cpu')
        y_a, y_b = y_a.to('cpu'), y_b.to('cpu')

        # 损失函数要在CPU算

        # 计算损失
        batch_loss = l_list * criterion(output, y_a) + (1 - l_list) * criterion(output, y_b)
        batch_loss = batch_loss.mean()
        batch_loss = batch_loss / accumulation_steps

        # 反向传播
        batch_loss.backward()

        train_loss += batch_loss.item()

        _, pred = torch.max(output.data, 1)
        total += y.shape[0]
        correct_a = pred.eq(y_a.data).to('cpu').sum()
        correct_b = pred.eq(y_b.data).to('cpu').sum()

        # correct += pred.eq(y.data).to('cpu').sum()
        correct += (l_list * correct_a + (1 - l_list) * correct_b).mean().item()
        
        # 多个 batch 更新一次参数
        if batch % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 输出一次结果
        if (batch+1) % 300 == 0 or batch == num_batches-1:
            current = (batch + 1) * batch_size if batch < num_batches -1 else size

            print(f"Loss: {train_loss/(batch + 1):>7f} | Acc: {correct/total*100:>4.2f}% [{correct:9.2f}/{total:>6d}] | Train [{current:>5d}/{size:>5d}] in {time.time() - start:>5f}s")
            start = time.time()

    del train_loss, batch_loss, output, mixed_X, X
