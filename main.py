import torch
import collections
import argparse
import dataloader
import test
import train
import utils
import time

def main(path, batch_size, epochs, learning_rate,
         momentum, weight_decay, optimize_after_batches, device, model_str, save_model):

    # checkpoint路径
    checkpoint = utils.get_checkpoint_path(model_str, device)

    # 选择模型
    model = utils.get_model(model_str, device)

    # 获取训练集和验证集
    train_dataloader = dataloader.get_dataloader(path=path, batch_size=batch_size, input_size=40, mode='Train')
    val_dataloader = dataloader.get_dataloader(path=path, batch_size=24, input_size=40, mode='Val')
    test_dataloader = dataloader.get_dataloader(path=path, batch_size=24, input_size=40, mode='Test')

    # 选择设备，N卡、DML、CPU
    device = utils.select_device(device)

    # Load the model on the device
    start = time.time()
    
    print('Finished moving {} to device: {} in {}s.'.format(model_str, device, time.time() - start))
 
    # 损失函数
    # label smoothing
    criterion = utils.LabelSmoothSoftmaxCEV1()
    # 交叉熵
    # criterion = nn.CrossEntropyLoss()

    # 定义优化器，SGD
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True)

    # 余弦退火，降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    # 校验集和训练集的准确率
    highest_val_accuracy = 0
    highest_test_accuracy = 0

    for t in range(epochs):
        print(f"\nEpoch {t+1}, Learning Rate {scheduler.get_last_lr()}\n-------------------------------")

        # Train
        train.train(train_dataloader,
              model,
              device,
              criterion,
              optimizer,
              optimize_after_batches)

        # Test
        val_accuracy = test.eval(val_dataloader,
                                model,
                                device,
                                criterion)

        test_accuracy = test.eval(test_dataloader,
                                model,
                                device,
                                criterion,
                                'Test')

        # 最高准确率
        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy

            print(f"current highest Acc: {(100*highest_val_accuracy):>0.1f}%")

            if save_model:

                state_dict = collections.OrderedDict()
                for key in model.state_dict().keys():
                    state_dict[key] = model.state_dict()[key].to("cpu")

                torch.save({
                    'epoch': t,
                    'model_state_dict': state_dict,
                    'epoch': t+1,
                    'best_acc': highest_val_accuracy
                }, checkpoint)

        if test_accuracy > highest_test_accuracy:
            highest_test_accuracy = test_accuracy

        # 更新学习率
        scheduler.step()

    print(f"Done! with highest_accuracy: Val {highest_val_accuracy:>0.3f} | Test {highest_test_accuracy:>0.3f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, default="fer2013.csv", help="Path to fer2013 dataset.")
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size to train with.')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='The number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, default=0.03, metavar='LR', help='The learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='The percentage of past parameters to store.')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='The parameter to decay weights.')
    parser.add_argument('--optimize_after_batches', default=16, type=int, help='After how many batches do optimize.')
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    parser.add_argument('--model', type=str, default='resnet18', help='The model to use.')
    parser.add_argument('--save_model', action='store_true', help='save model state_dict to file')
    args = parser.parse_args()

    print (args)
    main(args.path, args.batch_size, args.epochs, args.learning_rate,
         args.momentum, args.weight_decay, args.optimize_after_batches, args.device, args.model, args.save_model)