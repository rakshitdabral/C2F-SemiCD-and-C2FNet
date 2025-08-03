import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import rasterio
from PIL import Image, ImageEnhance

# ─── your existing augmentations ────────────────────────────────────────────

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
        arr[x, y] = 0 if random.random()<0.5 else 255
    return Image.fromarray(arr)

# ─── helper to load a .tif into a PIL image ────────────────────────────────

def load_tif_rgb(path, bands=[4,3,2]):
    """Read bands [4,3,2] from a GeoTIFF, replace NaNs, return as PIL RGB."""
    with rasterio.open(path) as src:
        arr = src.read(bands).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
    # transpose to HWC and clip/scale if you wish; here we normalize to [0,1]
    arr = np.clip(arr, 0, None)
    arr = arr / (arr.max() or 1)
    arr = np.transpose(arr, (1, 2, 0))
    # convert to 0–255 uint8 for PIL
    img8 = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(img8)

def load_tif_gray(path):
    """Read first band of a mask-tif, replace NaNs, return as PIL L."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
    # assume mask is binary or 0–1; scale up
    arr = np.clip(arr, 0, 1)
    img8 = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(img8, mode='L')

# ─── Paired (labelled) dataset ─────────────────────────────────────────────

class PairedChangeDataset(data.Dataset):
    def __init__(self, csv_path, trainsize, mosaic_ratio=0.75):
        df = pd.read_csv(csv_path)
        # Check column names and adapt to what's available in the CSV
        if 'past_image_path' in df.columns and 'present_image_path' in df.columns:
            self.past_paths = df['past_image_path'].tolist()
            self.present_paths = df['present_image_path'].tolist()
            if 'mask_path' in df.columns:
                self.mask_paths = df['mask_path'].tolist()
            else:
                self.mask_paths = df['mask'].tolist() if 'mask' in df.columns else []
        else:
            # Fallback to original column names
            self.past_paths = df['past'].tolist()
            self.present_paths = df['present'].tolist()
            self.mask_paths = df['mask'].tolist()
            
        self.trainsize = trainsize
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
            A, B   = colorEnhance(A, B)
            M      = randomPeper(M)
        else:
            # you can re-implement your mosaic logic here using
            # self._load_pair(random idx) four times...
            A, B, M = self._load_pair(idx)  # fallback to non-mosaic

        A = self.img_transform(A)
        B = self.img_transform(B)
        M = self.gt_transform(M)
        return A, B, M

    def _load_pair(self, idx):
        A = load_tif_rgb(self.past_paths[idx])
        B = load_tif_rgb(self.present_paths[idx])
        M = load_tif_gray(self.mask_paths[idx])
        return A, B, M

# ─── Unpaired (unlabelled) dataset ─────────────────────────────────────────

class UnpairedChangeDataset(data.Dataset):
    def __init__(self, csv_path, trainsize):
        df = pd.read_csv(csv_path)
        # Check column names and adapt to what's available in the CSV
        if 'past_image_path' in df.columns and 'present_image_path' in df.columns:
            self.past_paths = df['past_image_path'].tolist()
            self.present_paths = df['present_image_path'].tolist()
        else:
            # Fallback to original column names
            self.past_paths = df['past'].tolist()
            self.present_paths = df['present'].tolist()
            
        self.trainsize = trainsize

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.past_paths)

    def __getitem__(self, idx):
        A = load_tif_rgb(self.past_paths[idx])
        B = load_tif_rgb(self.present_paths[idx])
        A = self.img_transform(A)
        B = self.img_transform(B)
        return A, B

# ─── loader factories ───────────────────────────────────────────────────────

def get_paired_loader(csv_path, batchsize, trainsize,
                      mosaic_ratio=0.75,
                      num_workers=4, shuffle=True, pin_memory=True):
    ds = PairedChangeDataset(csv_path, trainsize, mosaic_ratio)
    return data.DataLoader(ds, batch_size=batchsize, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)

def get_unpaired_loader(csv_path, batchsize, trainsize,
                        num_workers=4, shuffle=True, pin_memory=True):
    ds = UnpairedChangeDataset(csv_path, trainsize)
    return data.DataLoader(ds, batch_size=batchsize, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)
