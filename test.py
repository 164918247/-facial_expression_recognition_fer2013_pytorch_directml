import torch
import utils
import time

def eval(dataloader, model, device, loss, mode='Val'):
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
            batch_loss = loss(output_avg, y)
            test_loss += batch_loss.item()

            _, pred = torch.max(output_avg.data, 1)
            correct += pred.eq(y.data).to('cpu').sum().item()

    # 计算总损失
    test_loss = test_loss / num_batches
    accuracy = correct / total

    print(f"{mode} Error: Acc {(100*accuracy):>0.1f}% | Avg loss: {test_loss:>8f} in {time.time() - start:>5f}s")

    del test_loss, batch_loss, output, X

    return accuracy