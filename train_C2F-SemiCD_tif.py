import time
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from utils import data_loader_tif  # Changed from data_loader to data_loader_tif
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from network.SemiModel import SemiModel

# Initialize start time for tracking total training time
start=time.time()

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


def train1(train_paired_loader, train_unpaired_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2,
           data_name, save_path, net, ema_net, criterion, semicriterion, optimizer, use_ema, num_epoches, epoch):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)
    ema_net.train(True)

    length = 0
    st = time.time()
    loss_semi = torch.zeros(1)
    
    # Process paired (labeled) data
    with tqdm(total=len(train_paired_loader), desc=f'Eps {epoch}/{num_epoches} - Paired', unit='img') as pbar:
        for i, (A, B, Y) in enumerate(train_paired_loader):
            A = A.cuda()
            B = B.cuda()
            Y = Y.cuda()
            with_label = torch.ones(A.size(0), dtype=torch.bool).cuda()  # All samples have labels

            optimizer.zero_grad()
            preds = net(A, B)
            loss = criterion(preds[0], Y) + criterion(preds[1], Y)

            loss.backward()
            optimizer.step()

            # Update EMA model
            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha=0.99)

            epoch_loss += loss.item()

            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_train.add_batch(target, pred)
            pbar.set_postfix(**{'LAll': loss.item(), 'LSemi': 0.0})
            pbar.update(1)
            length += 1
    
    # Process unpaired (unlabeled) data if using EMA
    if use_ema and train_unpaired_loader is not None:
        with tqdm(total=len(train_unpaired_loader), desc=f'Eps {epoch}/{num_epoches} - Unpaired', unit='img') as pbar:
            for i, (A, B) in enumerate(train_unpaired_loader):
                A = A.cuda()
                B = B.cuda()
                
                optimizer.zero_grad()
                preds = net(A, B)
                
                # Generate pseudo-labels with EMA model
                with torch.no_grad():
                    pseudo_attn, pseudo_preds = ema_net(A, B)
                    pseudo_attn, pseudo_preds = torch.sigmoid(pseudo_attn).detach(), torch.sigmoid(pseudo_preds).detach()
                
                loss_semi = semicriterion(preds[0], pseudo_attn) + semicriterion(preds[1], pseudo_preds)
                loss = 0.2 * loss_semi  # Semi-supervised coefficient 0.2
                
                loss.backward()
                optimizer.step()
                
                # Update EMA model
                with torch.no_grad():
                    update_ema_variables(net, ema_net, alpha=0.99)
                
                epoch_loss += loss.item()
                pbar.set_postfix(**{'LAll': loss.item(), 'LSemi': loss_semi.item()})
                pbar.update(1)
                length += 1

    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            IoU, Pre, Recall, F1))

    print("Start validating!")

    net.train(False)
    net.eval()
    ema_net.train(False)
    ema_net.eval()
    for i, (A, B, Y) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = Y.cuda()
            preds = net(A, B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)

            preds_ema = ema_net(A, B)[1]
            Eva_val2.add_batch(target, (preds_ema > 0).cpu().numpy().astype(int))
            length += 1

    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))

    print('[Ema Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
    Eva_val2.Intersection_over_Union()[1], Eva_val2.Precision()[1], Eva_val2.Recall()[1], Eva_val2.F1()[1]))
    new_iou = IoU[1]  # Store teacher model?
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        print('best_epoch', epoch)
        student_dir = save_path + '_train1_' + '_best_student_iou.pth'
        # Save model state
        student_state = {'best_student_net': net.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch}
        torch.save(student_state, student_dir)
        torch.save(ema_net.state_dict(),
                   save_path + '_train1_' + '_best_teacher_iou.pth')  # Save teacher model when student has best accuracy
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
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
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='TIF_Dataset',
                        help='dataset name')
    parser.add_argument('--model_name', type=str, default='SemiModel_TIF',
                        help='model name')
    parser.add_argument('--save_path', type=str, default='./output/C2F-SemiCD/TIF/')
    parser.add_argument('--paired_csv', type=str, default='./dataset_delhi/paired_dataset.csv',
                        help='CSV file with labeled data (mask, past, present)')
    parser.add_argument('--unpaired_csv', type=str, default='./dataset_delhi/unpaired_pairs.csv',
                        help='CSV file with unlabeled data (past, present)')
    parser.add_argument('--val_csv', type=str, default='./dataset_delhi/val_dataset.csv',
                        help='CSV file with validation data')

    opt = parser.parse_args()
    print(f'Using TIF files with {opt.train_ratio*100}% labeled data, semi-supervised loss coefficient: 0.2')

    # Set GPU device
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
        print(f'USE GPU {opt.gpu_id}')

    # Create save directory if it doesn't exist
    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # Load datasets using TIF loaders
    train_paired_loader = data_loader_tif.get_paired_loader(
        opt.paired_csv, opt.batchsize, opt.trainsize, 
        num_workers=8, shuffle=True, pin_memory=False
    )
    
    # Load unpaired (unlabeled) data if available
    train_unpaired_loader = None
    if os.path.exists(opt.unpaired_csv):
        train_unpaired_loader = data_loader_tif.get_unpaired_loader(
            opt.unpaired_csv, opt.batchsize, opt.trainsize,
            num_workers=8, shuffle=True, pin_memory=False
        )
    
    # Load validation data
    val_loader = None
    if os.path.exists(opt.val_csv):
        val_loader = data_loader_tif.get_paired_loader(
            opt.val_csv, opt.batchsize, opt.trainsize,
            num_workers=6, shuffle=False, pin_memory=False
        )
    else:
        # If no validation CSV is provided, use a portion of the paired data for validation
        print("No validation CSV found. Using 20% of paired data for validation.")
        # This is a simplified approach - in a real scenario, you might want to create a proper validation split
        from torch.utils.data import random_split
        from utils.data_loader_tif import PairedChangeDataset
        
        full_dataset = PairedChangeDataset(opt.paired_csv, opt.trainsize)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_paired_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batchsize, shuffle=True,
            num_workers=8, pin_memory=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batchsize, shuffle=False,
            num_workers=6, pin_memory=False
        )

    # Initialize evaluators
    Eva_train = Evaluator(num_class=2)
    Eva_train2 = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)

    # Initialize models
    model = SemiModel().cuda()
    ema_model = SemiModel().cuda()

    for param in ema_model.parameters():
        param.detach_()

    # Initialize loss functions and optimizer
    criterion = nn.BCEWithLogitsLoss().cuda()
    semicriterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start training...")

    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        # Start with supervised training for a few epochs, then add semi-supervised
        if epoch < 5:
            use_ema = False
        else:
            use_ema = True

        Eva_train.reset()
        Eva_train2.reset()
        Eva_val.reset()
        Eva_val2.reset()
        
        train1(train_paired_loader, train_unpaired_loader, val_loader, 
               Eva_train, Eva_train2, Eva_val, Eva_val2, 
               data_name, opt.save_path, model, ema_model, 
               criterion, semicriterion, optimizer, use_ema, opt.epoch, epoch)

        lr_scheduler.step()

    end = time.time()
    print('Total training time:', end - start)