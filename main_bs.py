'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np
import heapq
from models import *
from utils import progress_bar

import logging


#TODO 加入bs部分代码，带warning up / 需要调整epoch的大小，因为每轮的batchsize变小了

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="resnet18", type=str, help='choose which model to use')
parser.add_argument('--seed', default=1234, type=int, help='seed and experiment code')
parser.add_argument('--nepochs', default=300, type=int, help='how many epochs to run')
parser.add_argument('--bs', default=2, type=int, help='bs multi rate')
parser.add_argument('--nwarm', default=100, type=int, help='warming up epochs')
parser.add_argument('--select_function', default="entro", type=str, help='how to select data')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# experiment result
if not os.path.exists('./experiment/'+args.model):
    os.makedirs('./experiment/'+args.model)


# logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logfile = "bs_" + args.model + '_' + str(args.seed) + '_' + str(args.nepochs) + '_' +str(args.bs) + '_' +str(args.select_function)
handler = logging.FileHandler('./experiment/' +args.model + '/' + logfile + '.txt')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)

# 屏幕输出控制
console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)

# 记录实验数据
logger.info("---------experiment setting------------")
logger.info("learning rate: {}".format(args.lr))
logger.info("model:" + args.model)
logger.info("Seed: {}".format(args.seed))
logger.info("nepochs: {}".format(args.nepochs))
logger.info("bs: {}".format(args.bs))
logger.info("nwarm: {}".format(args.nwarm))
logger.info("select_function: {}".format(args.select_function))

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
trainloader_bs = torch.utils.data.DataLoader(
    trainset, batch_size=int(128*args.bs), shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.model == "resnet18":
    net = ResNet18()

if args.model == "resnet101":
    net = ResNet101()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
# scheduler = MultiStepLR(optimizer_fn, milestones=[150, 225, 270], gamma=0.1)



# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if epoch < args.nwarm:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # ---------backpropagation-----------
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # break
    else:
        for _ in range(int(args.bs)):
            # --------------------------
            # -----If set bs>1,then add more iter to feed
            for batch_idx, (inputs, targets) in enumerate(trainloader_bs):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                # -----------bs process----------------------
                # ---entro selection----
                if args.select_function == 'entro':
                    outputs_cpu = outputs.softmax(1).cpu().detach().numpy()
                    # print(outputs_cpu)
                    # print(outputs_cpu.shape)
                    # out_bs = []
                    entros = []
                    for i in range(outputs_cpu.shape[0]):
                        log_data = outputs_cpu[i] * np.log(abs(outputs_cpu[i]) + 1e-6)
                        # print(log_data)
                        entro = np.sum(log_data)
                        entro = entro.mean()
                        # print(entro)
                        entros.append(-entro)
                    max_n_index_list = list(map(entros.index, heapq.nlargest(128, entros)))
                    outputs = outputs[max_n_index_list, ]
                    # print(type(outputs))
                    targets = targets[max_n_index_list, ]
                    # print(type(targets))
                    # print(outputs.softmax(1))
                    # print(outputs.size)
                # -------maxmin selection------------
                elif args.select_function == 'maxmin':
                    outputs_cpu = outputs.softmax(1).cpu().detach().numpy()
                    maxps = []
                    for i in range(outputs_cpu.shape[0]):
                        maxp = np.max(outputs_cpu[i])
                        maxps.append(-maxp)
                    max_n_index_list = list(map(maxps.index, heapq.nlargest(128, maxps)))
                    outputs = outputs[max_n_index_list, ]
                    # print(type(outputs))
                    targets = targets[max_n_index_list, ]

                # ---------backpropagation-----------
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # break

    if epoch%10 == 0:
        logger.info("Train process")
        if epoch<args.nwarm:
            logger.info('epoch: {}  loss: {}  acc {} lr {}'.format(epoch,
                                                                   (train_loss/len(trainloader)), 100.*correct/total,
                                                                   optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            logger.info('epoch: {}  loss: {}  acc {} lr {}'.format(epoch,
                                                                   (train_loss/len(trainloader_bs)), 100.*correct/total,
                                                                   optimizer.state_dict()['param_groups'][0]['lr']))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Best acc on test renew' + "  " + str(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    if epoch%10 == 0:
        logger.info("Test process")
        logger.info('epoch: {}  acc: {}  best acc {}'.format(epoch, acc, best_acc))


for epoch in range(start_epoch, start_epoch+args.nepochs):
    train(epoch)
    test(epoch)
    scheduler.step()  # 阶段性调整学习率