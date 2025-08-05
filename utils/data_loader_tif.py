import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import rasterio
from PIL import Image, ImageEnhance
from functools import partial

# ─── Augmentation Functions ──────────────────────────────────────────────

def cv_random_flip(img_A, img_B, label):
    if random.random() > 0.5:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_A, img_B, label


def randomCrop(image_A, image_B, label, border=30):
    w, h = image_A.size
    cw = np.random.randint(w - border, w)
    ch = np.random.randint(h - border, h)
    region = ((w - cw)//2, (h - ch)//2, (w + cw)//2, (h + ch)//2)
    return image_A.crop(region), image_B.crop(region), label.crop(region)


def randomCrop_Mosaic(image_A, image_B, label, crop_w, crop_h):
    w, h = image_A.size
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return image_A.crop((x1, y1, x2, y2)), \
           image_B.crop((x1, y1, x2, y2)), \
           label.crop((x1, y1, x2, y2))


def randomRotation(image_A, image_B, label, angle_range=15):
    if random.random() > 0.8:
        ang = random.randint(-angle_range, angle_range)
        image_A = image_A.rotate(ang, Image.BICUBIC)
        image_B = image_B.rotate(ang, Image.BICUBIC)
        label   = label.rotate(ang, Image.NEAREST)
    return image_A, image_B, label


def colorEnhance(image_A, image_B):
    for fn in (ImageEnhance.Brightness, ImageEnhance.Contrast,
               ImageEnhance.Color, ImageEnhance.Sharpness):
        factor = random.uniform(0.5, 1.5)
        image_A = fn(image_A).enhance(factor)
        image_B = fn(image_B).enhance(factor)
    return image_A, image_B


def randomPeper(img):
    arr = np.array(img)
    n = int(0.0015 * arr.size)
    for _ in range(n):
        x = random.randrange(arr.shape[0])
        y = random.randrange(arr.shape[1])
        arr[x, y] = 0 if random.random() < 0.5 else 255
    return Image.fromarray(arr)

# ─── TIFF Loading Helpers ─────────────────────────────────────────────────

def load_tif_rgb(path, bands=[4,3,2]):
    try:
        with rasterio.open(path) as src:
            arr = src.read(bands).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0, None)
        mx = arr.max()
        if mx > 0: arr = arr / mx
        arr = np.transpose(arr, (1, 2, 0))
        img8 = (arr * 255).round().astype(np.uint8)
        return Image.fromarray(img8)
    except Exception as e:
        print(f"Error loading RGB TIFF {path}: {e}")
        return Image.fromarray(np.zeros((256,256,3), dtype=np.uint8))


def load_tif_gray(path):
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0, 1)
        img8 = (arr * 255).round().astype(np.uint8)
        return Image.fromarray(img8, mode='L')
    except Exception as e:
        print(f"Error loading Gray TIFF {path}: {e}")
        return Image.fromarray(np.zeros((256,256), dtype=np.uint8), mode='L')

# ─── Paired (Labelled) Dataset ────────────────────────────────────────────

class PairedChangeDataset(data.Dataset):
    def __init__(self, csv_path, trainsize, mosaic_ratio=0.75):
        df = pd.read_csv(csv_path)
        if 'past_image_path' in df.columns:
            self.past_paths    = df['past_image_path'].tolist()
            self.present_paths = df['present_image_path'].tolist()
            self.mask_paths    = df['mask_path'].tolist()
        else:
            self.past_paths    = df['past'].tolist()
            self.present_paths = df['present'].tolist()
            self.mask_paths    = df['mask'].tolist()

        self.trainsize    = trainsize
        self.mosaic_ratio = mosaic_ratio

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.past_paths)

    def __getitem__(self, idx):
        if random.random() > self.mosaic_ratio:
            A, B, M = self._load_pair(idx)
            A, B, M = cv_random_flip(A, B, M)
            A, B, M = randomCrop(A, B, M)
            A, B, M = randomRotation(A, B, M)
            A, B    = colorEnhance(A, B)
            M       = randomPeper(M)
        else:
            A, B, M = self._load_mosaic_pair(idx)
            A, B, M = cv_random_flip(A, B, M)
            A, B, M = randomRotation(A, B, M)
            A, B    = colorEnhance(A, B)
            M       = randomPeper(M)

        A = self.img_transform(A)
        B = self.img_transform(B)
        M = self.gt_transform(M)

        # ensure finite
        if not (torch.isfinite(A).all() and torch.isfinite(B).all() and torch.isfinite(M).all()):
            A = torch.nan_to_num(A)
            B = torch.nan_to_num(B)
            M = torch.nan_to_num(M)

        return A, B, M

    def _load_pair(self, idx):
        A = load_tif_rgb(self.past_paths[idx])
        B = load_tif_rgb(self.present_paths[idx])
        M = load_tif_gray(self.mask_paths[idx])
        return A, B, M

    def _load_mosaic_pair(self, idx):
        # pick 4 samples (self + 3 random)
        idxs = [idx] + [random.randint(0, len(self)-1) for _ in range(3)]
        imgs = [self._load_pair(i) for i in idxs]
        w = h = self.trainsize
        start_x, start_y = w//4, h//4
        offset_x = random.randint(start_x, w-start_x)
        offset_y = random.randint(start_y, h-start_y)
        sizes = [(offset_x, offset_y), (w-offset_x, offset_y),
                 (offset_x, h-offset_y), (w-offset_x, h-offset_y)]

        crops = []
        for (A0,B0,M0), (cw, ch) in zip(imgs, sizes):
            crop = randomCrop_Mosaic(A0.copy(), B0.copy(), M0.copy(), cw, ch)
            crops.append(crop)

        # assemble mosaic
        def to_np(img): return np.array(img)
        a1, b1, m1 = crops[0]; a2, b2, m2 = crops[1]
        a3, b3, m3 = crops[2]; a4, b4, m4 = crops[3]
        top_a = np.concatenate((to_np(a1), to_np(a2)), axis=1)
        bot_a = np.concatenate((to_np(a3), to_np(a4)), axis=1)
        img_a = np.concatenate((top_a, bot_a), axis=0)
        top_b = np.concatenate((to_np(b1), to_np(b2)), axis=1)
        bot_b = np.concatenate((to_np(b3), to_np(b4)), axis=1)
        img_b = np.concatenate((top_b, bot_b), axis=0)
        top_m = np.concatenate((to_np(m1), to_np(m2)), axis=1)
        bot_m = np.concatenate((to_np(m3), to_np(m4)), axis=1)
        mask  = np.concatenate((top_m, bot_m), axis=0)

        return Image.fromarray(img_a), Image.fromarray(img_b), Image.fromarray(mask)

# ─── Unpaired (Unlabelled) Dataset ───────────────────────────────────────

class UnpairedChangeDataset(data.Dataset):
    def __init__(self, csv_path, trainsize):
        df = pd.read_csv(csv_path)
        if 'past_image_path' in df.columns:
            self.past_paths    = df['past_image_path'].tolist()
            self.present_paths = df['present_image_path'].tolist()
        else:
            self.past_paths    = df['past'].tolist()
            self.present_paths = df['present'].tolist()

        self.trainsize = trainsize
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.past_paths)

    def __getitem__(self, idx):
        try:
            A = load_tif_rgb(self.past_paths[idx])
            B = load_tif_rgb(self.present_paths[idx])
            A = self.img_transform(A)
            B = self.img_transform(B)
            if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
                A = torch.nan_to_num(A)
                B = torch.nan_to_num(B)
            return A, B
        except Exception as e:
            print(f"Error processing unpaired idx={idx}: {e}")
            return torch.zeros(3, self.trainsize, self.trainsize), \
                   torch.zeros(3, self.trainsize, self.trainsize)

# ─── Test Dataset for Inference ────────────────────────────────────────────

class TestPairedChangeDataset(data.Dataset):
    def __init__(self, csv_path, trainsize):
        df = pd.read_csv(csv_path)
        if 'past_image_path' in df.columns:
            self.past_paths    = df['past_image_path'].tolist()
            self.present_paths = df['present_image_path'].tolist()
            self.mask_paths    = df['mask_path'].tolist()
        else:
            self.past_paths    = df['past'].tolist()
            self.present_paths = df['present'].tolist()
            self.mask_paths    = df['mask'].tolist()

        self.trainsize = trainsize
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.past_paths)

    def __getitem__(self, idx):
        A = load_tif_rgb(self.past_paths[idx])
        B = load_tif_rgb(self.present_paths[idx])
        M = load_tif_gray(self.mask_paths[idx])
        A = self.img_transform(A)
        B = self.img_transform(B)
        M = self.gt_transform(M)
        fname = os.path.basename(self.past_paths[idx])
        return A, B, M, fname

# ─── Collate Helpers ──────────────────────────────────────────────────────

def safe_collate_paired(batch, batchsize=None, trainsize=None):
    A, B, M = data.dataloader.default_collate(batch)
    A = A.contiguous(memory_format=torch.channels_last)
    B = B.contiguous(memory_format=torch.channels_last)
    return A, B, M


def safe_collate_unpaired(batch, batchsize=None, trainsize=None):
    A, B = data.dataloader.default_collate(batch)
    A = A.contiguous(memory_format=torch.channels_last)
    B = B.contiguous(memory_format=torch.channels_last)
    return A, B

# ─── DataLoader Factories ────────────────────────────────────────────────

def get_paired_loader(csv_path,
                      batchsize=4,
                      trainsize=256,
                      mosaic_ratio=0.75,
                      num_workers=2,
                      shuffle=True,
                      pin_memory=True,
                      train_ratio=1.0):
    ds = PairedChangeDataset(csv_path, trainsize, mosaic_ratio)
    
    # Apply train_ratio to use only a subset of the data
    if train_ratio < 1.0:
        total_samples = len(ds)
        subset_size = int(total_samples * train_ratio)
        # Use random indices to select subset
        indices = torch.randperm(total_samples)[:subset_size]
        ds = torch.utils.data.Subset(ds, indices)
        print(f"Using {len(ds)} out of {total_samples} samples ({train_ratio*100:.1f}%)")
    
    collate_fn = partial(safe_collate_paired,
                         batchsize=batchsize,
                         trainsize=trainsize)
    return data.DataLoader(ds,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=True,
                           prefetch_factor=2,
                           collate_fn=collate_fn)


def get_unpaired_loader(csv_path,
                        batchsize=4,
                        trainsize=256,
                        num_workers=2,
                        shuffle=True,
                        pin_memory=True):
    ds = UnpairedChangeDataset(csv_path, trainsize)
    collate_fn = partial(safe_collate_unpaired,
                         batchsize=batchsize,
                         trainsize=trainsize)
    return data.DataLoader(ds,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=True,
                           prefetch_factor=2,
                           collate_fn=collate_fn)


def get_test_loader(csv_path,
                    batchsize=4,
                    trainsize=256,
                    num_workers=2,
                    shuffle=False,
                    pin_memory=True):
    ds = TestPairedChangeDataset(csv_path, trainsize)
    return data.DataLoader(ds,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory)


if __name__ == "__main__":
    # sanity check
    paired_csv   = "labelled_dataset.csv"
    unpaired_csv = "unpaired_pairs.csv"
    paired_loader = get_paired_loader(paired_csv)
    unpaired_loader = get_unpaired_loader(unpaired_csv)
    for A, B, M in paired_loader:
        print("Paired batch shapes:", A.shape, B.shape, M.shape)
        break
    for A_u, B_u in unpaired_loader:
        print("Unpaired batch shapes:", A_u.shape, B_u.shape)
        break