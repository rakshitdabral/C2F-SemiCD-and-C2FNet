import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.visualization import Visualization
from utils.metrics import Evaluator
from network.SemiModel import SemiModel
import utils.data_loader_tif_combined as data_loader


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def update_ema_variables(student: nn.Module, teacher: nn.Module, alpha: float):
    for name, param in student.state_dict().items():
        teacher_param = teacher.state_dict()[name]
        teacher_param.mul_(alpha).add_(param, alpha=1-alpha)


def train_one_epoch(model, ema_model, loader,
                    criterion, semicriterion, optimizer,
                    use_ema, vis, Eva_sup, Eva_semi,
                    epoch, total_epochs, save_path):
    model.train()
    if use_ema:
        ema_model.train()

    epoch_loss = 0.0
    batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
    for A, B, M, has_label in pbar:
        A, B, M, has_label = A.cuda(), B.cuda(), M.cuda(), has_label.cuda()
        optimizer.zero_grad()

        attn_pred, mask_pred = model(A, B)

        # Supervised loss (only on labeled samples)
        if has_label.any():
            sup_preds = mask_pred[has_label]
            sup_targets = M[has_label]
            Ls = criterion(sup_preds, sup_targets)
        else:
            # zero-scalar with gradient
            Ls = mask_pred.sum() * 0.0

        # Semi-supervised consistency loss (only if EMA active)
        if use_ema and (~has_label).any():
            with torch.no_grad():
                t_attn, t_mask = ema_model(A[~has_label], B[~has_label])
                t_attn = torch.sigmoid(t_attn)
                t_mask = torch.sigmoid(t_mask)
            cs_attn = semicriterion(attn_pred[~has_label], t_attn)
            cs_mask = semicriterion(mask_pred[~has_label], t_mask)
            Lu = cs_attn + cs_mask
        else:
            Lu = mask_pred.sum() * 0.0

        loss = Ls + 0.2 * Lu
        loss.backward()
        optimizer.step()

        if use_ema:
            update_ema_variables(model, ema_model, alpha=0.99)

        # Metrics
        epoch_loss += loss.item()
        batches += 1
        bin_pred = (torch.sigmoid(mask_pred) > 0.5).long().cpu().numpy()
        bin_true = (M > 0.5).long().cpu().numpy()
        if has_label.any():
            Eva_sup.add_batch(bin_true[has_label.cpu().numpy()], bin_pred[has_label.cpu().numpy()])
        if (~has_label).any():
            Eva_semi.add_batch(bin_true[(~has_label).cpu().numpy()], bin_pred[(~has_label).cpu().numpy()])

        pbar.set_postfix(Loss=f"{loss.item():.4f}", Ls=f"{Ls.item():.4f}", Lu=f"{Lu.item():.4f}")

    # Logging
    IoU_sup = Eva_sup.Intersection_over_Union()[1]
    F1_sup = Eva_sup.F1()[1]
    vis.add_scalar(epoch, IoU_sup, 'mIoU')
    vis.add_scalar(epoch, F1_sup, 'F1')
    vis.add_scalar(epoch, epoch_loss/batches, 'train_loss')
    print(f"[Train] Epoch {epoch} Loss:{epoch_loss/batches:.4f} mIoU:{IoU_sup:.4f} F1:{F1_sup:.4f}")

    # Checkpoint
    global best_iou, best_epoch
    if IoU_sup >= best_iou:
        best_iou, best_epoch = IoU_sup, epoch
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()},
                   os.path.join(save_path, f"best_student_epoch{epoch}.pth"))
        torch.save(ema_model.state_dict(),
                   os.path.join(save_path, f"best_teacher_epoch{epoch}.pth"))
        print(f"New best mIoU {IoU_sup:.4f}, saved.")


def validate(model, ema_model, loader, Eva_val, Eva_val_ema):
    model.eval(); ema_model.eval()
    with torch.no_grad():
        for A, B, M, _ in tqdm(loader, desc="Val"):
            A, B, M = A.cuda(), B.cuda(), M.cuda()
            pred = (torch.sigmoid(model(A,B)[1]) > 0.5).long().cpu().numpy()
            true = (M > 0.5).long().cpu().numpy()
            Eva_val.add_batch(true, pred)
            pred_e = (torch.sigmoid(ema_model(A,B)[1]) > 0.5).long().cpu().numpy()
            Eva_val_ema.add_batch(true, pred_e)
    IoU, F1 = Eva_val.Intersection_over_Union()[1], Eva_val.F1()[1]
    IoUe, F1e = Eva_val_ema.Intersection_over_Union()[1], Eva_val_ema.F1()[1]
    print(f"[Val] Student IoU:{IoU:.4f} F1:{F1:.4f} | EMA IoU:{IoUe:.4f} F1:{F1e:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--train_csv', type=str,default="dataset_delhi/combined_dataset.csv")
    parser.add_argument('--val_csv', type=str, default="dataset_delhi/val_dataset.csv")
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--save_path', type=str, default="./output/")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_everything(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    best_iou, best_epoch = 0.0, 0

    train_loader = data_loader.get_csv_loader(
        args.train_csv, batchsize=args.batchsize, trainsize=args.trainsize,
        num_workers=8, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_csv_loader(
        args.val_csv, batchsize=args.batchsize, trainsize=args.trainsize,
        num_workers=4, shuffle=False, pin_memory=True)

    Eva_sup, Eva_semi = Evaluator(2), Evaluator(2)
    Eva_val, Eva_val_e = Evaluator(2), Evaluator(2)
    vis = Visualization(); vis.create_summary("SemiCD_TIF")

    model, ema_model = SemiModel().cuda(), SemiModel().cuda()
    for p in ema_model.parameters(): p.requires_grad_(False)
    criterion = nn.BCEWithLogitsLoss().cuda()
    semicriterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2.5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    for epoch in range(1, args.epoch+1):
        use_ema = (epoch > 5)
        Eva_sup.reset(); Eva_semi.reset(); Eva_val.reset(); Eva_val_e.reset()

        train_one_epoch(model, ema_model, train_loader,
                        criterion, semicriterion, optimizer,
                        use_ema, vis, Eva_sup, Eva_semi,
                        epoch, args.epoch, args.save_path)
        scheduler.step()
        validate(model, ema_model, val_loader, Eva_val, Eva_val_e)

    print(f"Done! Best mIoU {best_iou:.4f} at epoch {best_epoch}")
