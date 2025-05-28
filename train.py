# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import optim
import timeit
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from utils.utils import setup_seed, netParams
from utils.metric.metric import get_iou

from utils.losses.loss import CrossEntropyLoss2d,  \
    ProbOhemCrossEntropy2d, FocalLoss2d
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR, WarmupCosineLR

# event
from dataset.event.base_trainer import BaseTrainer
from datetime import datetime
import json

from spikingjelly.activation_based import functional



torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
print(torch_ver)

GLOBAL_SEED = 1234
torch.autograd.set_detect_anomaly(True)

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def parse_args():
    parser = ArgumentParser(
        description='SEINet: Lightweight Spike-driven Event-Image Fusion Network for Accurate and Efficient Semantic Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="SLTNet", help="model name")
    parser.add_argument('--dataset', type=str, default="DSEC_events", help="dataset: DSEC_events or DDD17_events")
    parser.add_argument('--input_size', type=str, default="480,640", help="DDD17_events:200,346,DSEC_events:480,640")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=11,
                        help="the number of classes in the dataset. 6 and 11 for ddd17 and dsec, respectively")
    parser.add_argument('--dataset_path', type=str, default="your onw dataset path",)
    parser.add_argument('--split', type=str, default="train", help="spilt in ['train', 'test', 'valid']")
    parser.add_argument('--nr_events_data', type=int, default=1)
    parser.add_argument('--delta_t_per_data', type=int, default=50)
    parser.add_argument('--nr_events_window', type=int, default=100000, help='DDD17:32000,DSEC:100000')
    parser.add_argument('--data_augmentation_train', type=bool, default=False)
    parser.add_argument('--event_representation', type=str, default="voxel_grid")
    parser.add_argument('--nr_temporal_bins', type=int, default=5)
    parser.add_argument('--require_paired_data_train', type=bool, default=True)
    parser.add_argument('--require_paired_data_val', type=bool, default=True)
    parser.add_argument('--separate_pol', type=bool, default=False)
    parser.add_argument('--normalize_event', type=bool, default=True)
    parser.add_argument('--fixed_duration', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="the batch size is set to 64 for 1 GPUs")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'radam', 'ranger'],
                        help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='StepLR', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9, help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true', default=False,
                        help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False,
                        help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', type=bool, default=False, help=' FocalLoss2d for cityscapes dataset')
    # datasets
    parser.add_argument('--use_ohem', type=bool, default=True, help='OhemCrossEntropy2d Loss for event dataset')
    parser.add_argument("--use_earlyloss", type=bool, default=True,
                        help='Use early-surpervised training for event dataset')
    parser.add_argument("--balance_weights", type=list, default=[1.0, 0.4], help='balance between out and early_out')

    # cuda setting
    parser.add_argument('--gpus', type=str, default="3")
    parser.add_argument('--workers', type=int, default=8)

    # checkpoint and log
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--arguFile', default="arguments.txt", help="storing the training arguments")

    args = parser.parse_args()

    return args


def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))

    print(args)

    # set the seed
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> building network")

    # build the model and initialization
    model = build_model(args.model, num_classes=args.classes, ohem=args.use_ohem, early_loss=args.use_earlyloss)
    functional.set_step_mode(model, step_mode='m')  # 将模型的前向传播和反向传播模式从默认的单步模式切换到多步模式,通常用于snn


    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))


    # DDD17/DSEC datasets
    base_trainer_instance = BaseTrainer()
    trainLoader, valLoader = base_trainer_instance.createDataLoaders(args)

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter

    if args.dataset in ['DDD17_events', 'DSEC_events']:
        if args.use_ohem:
            min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
            criteria = ProbOhemCrossEntropy2d(use_weight=False, ignore_label=ignore_label, thresh=0.7,
                                              min_kept=min_kept, balance_weights=args.balance_weights)
        elif args.use_focal:
            criteria = FocalLoss2d(weight=None, ignore_index=ignore_label, balance_weights=args.balance_weights)
        else:
            criteria = CrossEntropyLoss2d(weight=None, ignore_label=ignore_label)

    if torch.cuda.is_available():
        criteria = criteria.cuda(device)
        model = model.cuda(device)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.savedir = (args.savedir + args.dataset + '/' + args.model + "_" + current_time + "_" + str(args.split) + '/')

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0
    best_mIOU_per_class = []
    best_mIOU_val = 0
    best_acc_val = 0

    mIOU_val_list = []
    lossTr_list = []
    lossVal_list = []

    # continue training 如果训练中断，恢复训练
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_mIOU_val = checkpoint['best_miou']
            best_mIOU_per_class = checkpoint['best_miou_class']
            best_acc_val = checkpoint['best_acc']
            mIOU_val_list = checkpoint['mIOU_val_list']
            lossTr_list = checkpoint['lossTr_list']
            lossVal_list = checkpoint['lossVal_list']
            model.load_state_dict(checkpoint['model'])
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))
    logger.flush()

    # 记录参数
    arguFileLoc = args.savedir + args.arguFile
    if os.path.isfile(arguFileLoc):
        logger_argu = open(arguFileLoc, 'a')
    else:
        logger_argu = open(arguFileLoc, 'w')
        json.dump(args.__dict__, logger_argu, indent=2)
    logger_argu.flush()

    # tensorboard记录loss和iou曲线
    # writer = SummaryWriter(log_dir=args.savedir)

    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.90, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'ranger':
        optimizer = Ranger(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.95, 0.999), eps=1e-08,
            weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)

    # learming scheduling
    if args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'warmpoly':
        scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.8)
    elif args.lr_schedule == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.92, last_epoch=-1, verbose='deprecated')
    elif args.lr_schedule == 'warmupcosine':
        scheduler = WarmupCosineLR(optimizer, args.max_epochs * args.max_iter)

    epoches = []

    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training

        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, scheduler, epoch)
        lossTr_list.append(lossTr)

        if epoch < args.max_epochs:  # 每个epoch都验证且记录 记录mIou在.txt文件中
            epoches.append(epoch)
            mIOU_val, per_class_iu, acc_val, lossVal = val(args, valLoader, model, criteria)
            mIOU_val_list.append(mIOU_val)
            lossVal_list.append(lossVal)

            if mIOU_val > best_mIOU_val:
                best_mIOU_val = mIOU_val
                best_mIOU_per_class = per_class_iu

                # save the best miou model 
                model_file_name = args.savedir + '/model_best_miou.pth'
                state = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_miou": best_mIOU_val,
                    "best_acc": best_acc_val,
                    "best_miou_class": best_mIOU_per_class,
                    "mIOU_val_list": mIOU_val_list,
                    "lossTr_list": lossTr_list,
                    "lossVal_list": lossVal_list}
                torch.save(state, model_file_name)

            if acc_val > best_acc_val:
                best_acc_val = acc_val

                # save the best acc model 
                model_file_name = args.savedir + '/model_best_acc.pth'
                state = {"epoch": epoch + 1, "model": model.state_dict()}
                torch.save(state, model_file_name)

            # save the current model
            model_file_name = args.savedir + '/model_latest.pth'
            state = {"epoch": epoch + 1, "model": model.state_dict()}
            torch.save(state, model_file_name)

            # record train information
            logger.write(
                "\nEpoch No.: %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t best_mIOU_val(val) = %.4f\t best_mIOU_per_class(val) = [%s]\t best_acc(val) = %.4f\t lr= %.6f \n" % (
                    epoch, lossTr, mIOU_val, best_mIOU_val, best_mIOU_per_class, best_acc_val, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print(
                "Epoch No.: %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t best_mIOU_val(val) = %.4f\t best_mIOU_per_class(val) = [%s]\t best_acc(val) = %.4f\t lr= %.6f \n" % (
                    epoch,
                    lossTr,
                    mIOU_val,
                    best_mIOU_val,
                    best_mIOU_per_class,
                    best_acc_val,
                    lr))

        else:
            # record train information  #其他不用记录mIou
            logger.write("\n%d\t\t%.4f\t\t\t\t%.7f" % (epoch, lossTr, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))

        if epoch % 5 == 0 or epoch == (args.max_epochs - 1):
            # Plot the figures per 5 epochs
            # train loss
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")

            plt.savefig(args.savedir + "loss_vs_epochs_train.png")

            plt.clf()

            # val loss
            fig3, ax3 = plt.subplots(figsize=(11, 8))

            ax3.plot(range(start_epoch, epoch + 1), lossVal_list)
            ax3.set_title("Average validation loss vs epochs")
            ax3.set_xlabel("Epochs")
            ax3.set_ylabel("Current loss")

            plt.savefig(args.savedir + "loss_vs_epochs_val.png")

            plt.clf()

            # val miou
            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(epoches, mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            plt.legend(loc='lower right')

            plt.savefig(args.savedir + "iou_vs_epochs_val.png")

            plt.close('all')

    logger.close()


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """

    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)
    st = time.time()
    for iteration, batch in enumerate(train_loader, 0):
        if iteration == 1:
            break

        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration

        start_time = time.time()
        events, labels, _, RGBs = batch

        if torch_ver == '0.3':
            events = Variable(events).cuda(device)
            labels = Variable(labels.long()).cuda(device)
            RGBs = Variable(RGBs).cuda(device)
        else:
            events = events.cuda(device)
            labels = labels.long().cuda(device)
            RGBs = RGBs.cuda(device)

        output = model(events, RGBs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_loss.append(loss[0].item())
        time_taken = time.time() - start_time

        lr = optimizer.param_groups[0]['lr']

        if iteration % 10 == 0:
            if args.use_ohem:
                print(
                    '=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f ohem_loss: %.3f early_loss: %.3f time:%.2f' % (
                        epoch + 1, args.max_epochs,
                        iteration + 1, total_batches,
                        lr, loss[0].item(), loss[1].item(), loss[2].item(), time_taken))
            else:
                print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (
                    epoch + 1, args.max_epochs,
                    iteration + 1, total_batches,
                    lr, loss[0].item(), time_taken))
    scheduler.step()  # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def val(args, val_loader, model, criterion):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    epoch_loss = []
    data_list = []
    for i, (event, label, _, RGB) in enumerate(val_loader):
        start_time = time.time()
        with torch.no_grad():
            event_var = event.cuda(device)
            label_var = label.long().cuda(device)
            RGB = RGB.permute(0, 3, 1, 2)
            RGB_var = RGB.cuda(device)
            output = model(event_var,RGB_var)
        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        loss = criterion(output, label_var)
        epoch_loss.append(loss[0].item())

        if len(output) == 2:
            output = output[0]
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu, acc = get_iou(data_list, args.classes)
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    return meanIoU, per_class_iu, acc, average_epoch_loss_val


if __name__ == '__main__':

    start = timeit.default_timer()
    args = parse_args()

    if args.dataset == 'DDD17_events':
        args.classes = 6
        args.input_size = '200,346'
        ignore_label = 255
    elif args.dataset == 'DSEC_events':
        args.classes = 11
        args.input_size = '440,640'
        ignore_label = 255
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
