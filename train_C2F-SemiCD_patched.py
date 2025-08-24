
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from utils import data_loader
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from network.SemiModel import SemiModel

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def update_ema_variables(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2,
           data_name, save_path, net, ema_net, criterion, semicriterion, optimizer, use_ema, num_epoches, epoch):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0.0
    net.train(True)
    ema_net.train(True)

    length = 0  # number of effective optimization steps
    loss_semi_display = 0.0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epoches}', unit='batch') as pbar:
        for i, (A, B, mask, with_label) in enumerate(train_loader):
            A = A.cuda(non_blocking=True)
            B = B.cuda(non_blocking=True)
            Y = mask.cuda(non_blocking=True)
            with_label = with_label.cuda(non_blocking=True)

            # binarize masks robustly (avoid soft labels from aug/interp)
            Y = (Y > 0.5).float()

            optimizer.zero_grad(set_to_none=True)

            if not use_ema:
                # warmup: learn only from labeled patches; still advance the pbar for skipped batches
                if with_label.any():
                    preds = net(A[with_label], B[with_label])
                    loss = criterion(preds[0], Y[with_label]) + criterion(preds[1], Y[with_label])

                    out = torch.sigmoid(preds[1])
                    pred_bin = (out >= 0.5).float()

                    # metrics on labeled subset
                    Eva_train.add_batch(Y[with_label].detach().cpu().numpy().astype(int),
                                        pred_bin.detach().cpu().numpy().astype(int))

                    epoch_loss += float(loss.item())
                    length += 1

                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        update_ema_variables(net, ema_net, alpha=0.99)

                    loss_semi_display = 0.0
                # always tick the bar for each batch
                pbar.set_postfix(LAll=float(epoch_loss/(length if length>0 else 1)), LSemi=float(loss_semi_display))
                pbar.update(1)
                continue

            # ---- Semi-supervised (EMA) branch ----
            preds = net(A, B)

            # supervised part (may be zero if no labeled in batch)
            if with_label.any():
                sup = criterion(preds[0][with_label], Y[with_label]) + criterion(preds[1][with_label], Y[with_label])
            else:
                sup = torch.tensor(0.0, device=A.device)

            # unsupervised part on "unlabeled" indices
            if (~with_label).any():
                with torch.no_grad():
                    z1 = A[~with_label]
                    z2 = B[~with_label]
                    pa, pp = ema_net(z1, z2)
                    pa, pp = torch.sigmoid(pa).detach(), torch.sigmoid(pp).detach()
                loss_semi = semicriterion(preds[0][~with_label], pa) + semicriterion(preds[1][~with_label], pp)
                loss_semi_display = float(loss_semi.item())
            else:
                loss_semi = torch.tensor(0.0, device=A.device)
                loss_semi_display = 0.0

            loss = sup + 0.2 * loss_semi

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha=0.99)

            epoch_loss += float(loss.item())
            length += 1

            # predictions for metrics
            out_all = torch.sigmoid(preds[1])
            pred_bin_all = (out_all >= 0.5).float()

            # training metrics: labeled -> Eva_train, unlabeled -> Eva_train2 (uses GT for monitoring only)
            if with_label.any():
                Eva_train.add_batch(Y[with_label].detach().cpu().numpy().astype(int),
                                    pred_bin_all[with_label].detach().cpu().numpy().astype(int))
            if (~with_label).any():
                Eva_train2.add_batch(Y[~with_label].detach().cpu().numpy().astype(int),
                                     pred_bin_all[~with_label].detach().cpu().numpy().astype(int))

            pbar.set_postfix(LAll=float(loss.item()), LSemi=float(loss_semi_display))
            pbar.update(1)

    # aggregate epoch metrics
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / max(1, length)

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print('Epoch [%d/%d], Loss: %.4f,' % (epoch, num_epoches, train_loss))
    print('[Training Labeled] IoU: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f' % (IoU, Pre, Recall, F1))

    if use_ema:
        print('[Training Unlabeled (monitoring)] IoU: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f' % (
            Eva_train2.Intersection_over_Union()[1], Eva_train2.Precision()[1],
            Eva_train2.Recall()[1], Eva_train2.F1()[1]))

    print("Start validating!")

    # ---- Validation ----
    net.eval()
    ema_net.eval()

    with torch.no_grad():
        for i, (A, B, mask, filename) in enumerate(tqdm(val_loader, desc='Valid', unit='batch')):
            A = A.cuda(non_blocking=True)
            B = B.cuda(non_blocking=True)
            Y = (mask.cuda(non_blocking=True) > 0.5).float()

            # student
            logits = net(A, B)[1]
            pred = (torch.sigmoid(logits) >= 0.5).float()
            Eva_val.add_batch(Y.detach().cpu().numpy().astype(int),
                              pred.detach().cpu().numpy().astype(int))

            # teacher
            logits_ema = ema_net(A, B)[1]
            pred_ema = (logits_ema >= 0).float()  # same as sigmoid >= 0.5
            Eva_val2.add_batch(Y.detach().cpu().numpy().astype(int),
                               pred_ema.detach().cpu().numpy().astype(int))

    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    print('[EMA Validation] IoU: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f' % (
        Eva_val2.Intersection_over_Union()[1], Eva_val2.Precision()[1],
        Eva_val2.Recall()[1], Eva_val2.F1()[1]))

    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        print('Best Model IoU: %.4f; F1: %.4f; Best epoch: %d' % (IoU[1], F1[1], best_epoch))

        student_dir = save_path + '_train1_' + '_best_student_iou.pth'
        student_state = {
            'best_student_net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        if not os.path.exists(os.path.dirname(student_dir)):
            os.makedirs(os.path.dirname(student_dir), exist_ok=True)
        torch.save(student_state, student_dir)
        torch.save(ema_net.state_dict(), save_path + '_train1_' + '_best_teacher_iou.pth')
    print('Best Model IoU :%.4f; (current F1: %.4f)' % (best_iou, F1[1]))
    vis.close_summary()


if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--train_ratio', type=float, default=0.05, help='Proportion of the labeled images')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='WHU', help='dataset name')
    parser.add_argument('--model_name', type=str, default='SemiModel_noema04', help='model name')
    parser.add_argument('--save_path', type=str, default='./output/C2F-SemiCD/WHU-5/')
    # parser.add_argument('--save_path', type=str, default='./output/C2FNet/WHU/')

    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = '/data/chengxi.han/data/LEVIR CD Dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/LEVIR CD Dataset256/val/'
    elif opt.data_name == 'WHU':
        # Change these paths to use your local dataset
        opt.train_root = './labelled/train/'
        opt.val_root = './labelled/val/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/train/'
        opt.val_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/val/'
    elif opt.data_name == 'DSIFN':
        opt.train_root = '/data/chengxi.han/data/DSIFN256/train/'
        opt.val_root = '/data/chengxi.han/data/DSIFN256/val/'
    elif opt.data_name == 'SYSU':
        opt.train_root = '/data/chengxi.han/data/SYSU-CD/train/'
        opt.val_root = '/data/chengxi.han/data/SYSU-CD/val/'
    elif opt.data_name == 'S2Looking':
        opt.train_root = '/data/chengxi.han/data/S2Looking256/train/'
        opt.val_root = '/data/chengxi.han/data/S2Looking256/val/'
    elif opt.data_name == 'GoogleGZ':
        opt.train_root = '/data/chengxi.han/data/Google_GZ_CD256/train/'
        opt.val_root = '/data/chengxi.han/data/Google_GZ_CD256/val/'
    elif opt.data_name == 'LEVIRsup-WHUunsup':
        opt.train_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/train/'
        opt.val_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/val/'
    elif opt.data_name == 'LABELLED':
        opt.train_root = './labelled/train/'
        opt.val_root = './labelled/val/'

    train_loader = data_loader.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize, opt.train_ratio, num_workers=8, shuffle=True, pin_memory=False)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=6, shuffle=False, pin_memory=False)

    Eva_train = Evaluator(num_class=2)
    Eva_train2 = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)

    model = SemiModel().cuda()
    ema_model = SemiModel().cuda()

    for param in ema_model.parameters():
        param.detach_()

    criterion = nn.BCEWithLogitsLoss().cuda()
    semicriterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")

    # inclusive range to actually run 'epoch' epochs
    for epoch in range(1, opt.epoch + 1):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        # warmup epochs: labeled-only
        use_ema = epoch >= 5

        Eva_train.reset()
        Eva_train2.reset()
        Eva_val.reset()
        Eva_val2.reset()

        train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2, data_name, save_path, model,
               ema_model, criterion, semicriterion, optimizer, use_ema, opt.epoch, epoch)

        lr_scheduler.step()
