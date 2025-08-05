import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import rasterio
from PIL import Image, ImageEnhance


def load_tif_rgb(path, bands=[4, 3, 2], trainsize=None):
    """Load a multi-band TIFF as PIL RGB, normalize and scale."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(bands).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0, None)
        mx = arr.max()
        if mx > 0:
            arr = arr / mx
        arr = np.transpose(arr, (1, 2, 0))
        img8 = (arr * 255).round().astype(np.uint8)
        return Image.fromarray(img8)
    except Exception:
        size = trainsize or 256
        return Image.new('RGB', (size, size))


def load_tif_gray(path, trainsize=None):
    """Load a single-band TIFF as PIL grayscale, clip to [0,1]."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0, 1)
        img8 = (arr * 255).round().astype(np.uint8)
        return Image.fromarray(img8, mode='L')
    except Exception:
        size = trainsize or 256
        return Image.new('L', (size, size))


# ─── Augmentations ─────────────────────────────────────────────────────────
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
    x1, y1 = (w - crop_w)//2, (h - crop_h)//2
    x2, y2 = x1 + crop_w, y1 + crop_h
    return (image_A.crop((x1, y1, x2, y2)),
            image_B.crop((x1, y1, x2, y2)),
            label.crop((x1, y1, x2, y2)))


def randomRotation(image_A, image_B, label, angle_range=15):
    if random.random() > 0.8:
        ang = random.randint(-angle_range, angle_range)
        image_A = image_A.rotate(ang, Image.BICUBIC)
        image_B = image_B.rotate(ang, Image.BICUBIC)
        label = label.rotate(ang, Image.NEAREST)
    return image_A, image_B, label


def colorEnhance(image_A, image_B):
    for fn in (ImageEnhance.Brightness, ImageEnhance.Contrast,
               ImageEnhance.Color, ImageEnhance.Sharpness):
        factor = random.uniform(0.5, 1.5)
        image_A = fn(image_A).enhance(factor)
        image_B = fn(image_B).enhance(factor)
    return image_A, image_B


def randomGaussian(image, mean=0.1, sigma=0.35):
    arr = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(mean, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 1)
    img8 = (arr * 255).astype(np.uint8)
    return Image.fromarray(img8)


def randomPeper(img):
    arr = np.array(img)
    n = int(0.0015 * arr.size)
    for _ in range(n):
        x = random.randrange(arr.shape[0])
        y = random.randrange(arr.shape[1])
        arr[x, y] = 0 if random.random() < 0.5 else 255
    return Image.fromarray(arr)


class SemiCSVChangeDataset(data.Dataset):
    """
    Single dataset for semi-supervised training from CSV.
    CSV must have columns: ['past' or 'past_image_path'], ['present' or 'present_image_path'],
    optional ['mask' or 'mask_path'], optional ['with_label'].
    """
    def __init__(self, csv_path, trainsize, mosaic_ratio=0.75):
        df = pd.read_csv(csv_path)
        # column detection
        if 'past_image_path' in df.columns:
            self.past_paths = df['past_image_path'].tolist()
        elif 'past' in df.columns:
            self.past_paths = df['past'].tolist()
        else:
            raise KeyError("CSV must contain 'past_image_path' or 'past' column")
        if 'present_image_path' in df.columns:
            self.present_paths = df['present_image_path'].tolist()
        elif 'present' in df.columns:
            self.present_paths = df['present'].tolist()
        else:
            raise KeyError("CSV must contain 'present_image_path' or 'present' column")
        if 'mask_path' in df.columns:
            self.mask_paths = df['mask_path'].tolist()
        elif 'mask' in df.columns:
            self.mask_paths = df['mask'].tolist()
        else:
            self.mask_paths = [None] * len(self.past_paths)
        # label flags
        if 'with_label' in df.columns:
            self.flags = df['with_label'].astype(str).str.upper().eq('TRUE').tolist()
        else:
            self.flags = [m is not None and str(m).upper() != 'N/A' and not pd.isna(m)
                          for m in self.mask_paths]
        self.trainsize = trainsize
        self.mosaic_ratio = mosaic_ratio
        # filter invalid rows
        self._filter_rows()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def _filter_rows(self):
        keep = []
        for i, (p, q, m, f) in enumerate(zip(self.past_paths,
                                              self.present_paths,
                                              self.mask_paths,
                                              self.flags)):
            if not (os.path.exists(p) and os.path.exists(q)):
                continue
            try:
                with rasterio.open(p) as sp, rasterio.open(q) as sq:
                    if sp.shape != sq.shape:
                        continue
                if f:
                    if m is None or pd.isna(m) or str(m).upper() == 'N/A' or not os.path.exists(m):
                        continue
                    with rasterio.open(m) as sm:
                        if sm.shape != sp.shape:
                            continue
            except Exception:
                continue
            keep.append(i)
        self.past_paths = [self.past_paths[i] for i in keep]
        self.present_paths = [self.present_paths[i] for i in keep]
        self.mask_paths = [self.mask_paths[i] for i in keep]
        self.flags = [self.flags[i] for i in keep]

    def __len__(self):
        return len(self.past_paths)

    def _load_pair(self, idx):
        A = load_tif_rgb(self.past_paths[idx], trainsize=self.trainsize)
        B = load_tif_rgb(self.present_paths[idx], trainsize=self.trainsize)
        if self.flags[idx]:
            M = load_tif_gray(self.mask_paths[idx], trainsize=self.trainsize)
        else:
            M = Image.new('L', (self.trainsize, self.trainsize))
        return A, B, M

    def __getitem__(self, idx):
        if random.random() > self.mosaic_ratio:
            A, B, M = self._load_pair(idx)
            A, B, M = cv_random_flip(A, B, M)
            A, B, M = randomCrop(A, B, M)
            A, B, M = randomRotation(A, B, M)
            A, B = colorEnhance(A, B)
            M = randomPeper(M)
            if random.random() < 0.3:
                A = randomGaussian(A)
                B = randomGaussian(B)
        else:
            idxs = [idx] + [random.randint(0, len(self)-1) for _ in range(3)]
            samples = [self._load_pair(i) for i in idxs]
            w = h = self.trainsize
            ox = random.randint(w//4, w - w//4)
            oy = random.randint(h//4, h - h//4)
            sizes = [(ox, oy), (w-ox, oy), (ox, h-oy), (w-ox, h-oy)]
            crops = [randomCrop_Mosaic(a, b, m, cw, ch)
                     for (a,b,m), (cw,ch) in zip(samples, sizes)]
            to_np = lambda im: np.array(im)
            a1,b1,m1 = crops[0]; a2,b2,m2 = crops[1]
            a3,b3,m3 = crops[2]; a4,b4,m4 = crops[3]
            top_a = np.concatenate((to_np(a1), to_np(a2)), axis=1)
            bot_a = np.concatenate((to_np(a3), to_np(a4)), axis=1)
            top_b = np.concatenate((to_np(b1), to_np(b2)), axis=1)
            bot_b = np.concatenate((to_np(b3), to_np(b4)), axis=1)
            top_m = np.concatenate((to_np(m1), to_np(m2)), axis=1)
            bot_m = np.concatenate((to_np(m3), to_np(m4)), axis=1)
            A = Image.fromarray(np.concatenate((top_a, bot_a), axis=0))
            B = Image.fromarray(np.concatenate((top_b, bot_b), axis=0))
            M = Image.fromarray(np.concatenate((top_m, bot_m), axis=0))

        A = self.img_transform(A)
        B = self.img_transform(B)
        M = self.gt_transform(M)
        return A, B, M, self.flags[idx]


def safe_collate_csv(batch):
    As, Bs, Ms, flags = zip(*batch)
    As = torch.stack(As)
    Bs = torch.stack(Bs)
    Ms = torch.stack(Ms)
    flags = torch.tensor(flags, dtype=torch.bool)
    return As.contiguous(memory_format=torch.channels_last), \
           Bs.contiguous(memory_format=torch.channels_last), \
           Ms, flags


def get_csv_loader(csv_path,
                   batchsize=4,
                   trainsize=256,
                   mosaic_ratio=0.75,
                   num_workers=2,
                   shuffle=True,
                   pin_memory=True):
    ds = SemiCSVChangeDataset(csv_path, trainsize, mosaic_ratio)
    return data.DataLoader(ds,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=True,
                           prefetch_factor=2,
                           collate_fn=safe_collate_csv)


class TestCSVChangeDataset(data.Dataset):
    """Test loader over CSV: returns (A, B, M, flag, id)"""
    def __init__(self, csv_path, trainsize):
        df = pd.read_csv(csv_path)
        # column detection
        if 'past_image_path' in df.columns:
            self.past_paths = df['past_image_path'].tolist()
        elif 'past' in df.columns:
            self.past_paths = df['past'].tolist()
        else:
            raise KeyError("CSV must contain 'past_image_path' or 'past' column")
        if 'present_image_path' in df.columns:
            self.present_paths = df['present_image_path'].tolist()
        elif 'present' in df.columns:
            self.present_paths = df['present'].tolist()
        else:
            raise KeyError("CSV must contain 'present_image_path' or 'present' column")
        if 'mask_path' in df.columns:
            self.mask_paths = df['mask_path'].tolist()
        elif 'mask' in df.columns:
            self.mask_paths = df['mask'].tolist()
        else:
            self.mask_paths = [None] * len(self.past_paths)
        if 'with_label' in df.columns:
            self.flags = df['with_label'].astype(str).str.upper().eq('TRUE').tolist()
        else:
            self.flags = [m is not None and str(m).upper() != 'N/A' and not pd.isna(m)
                          for m in self.mask_paths]
        self.trainsize = trainsize
        SemiCSVChangeDataset._filter_rows(self)
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in self.present_paths]
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
        A = load_tif_rgb(self.past_paths[idx], trainsize=self.trainsize)
        B = load_tif_rgb(self.present_paths[idx], trainsize=self.trainsize)
        if self.flags[idx]:
            M = load_tif_gray(self.mask_paths[idx], trainsize=self.trainsize)
        else:
            M = Image.new('L', (self.trainsize, self.trainsize))
        A = self.img_transform(A)
        B = self.img_transform(B)
        M = self.gt_transform(M)
        return A, B, M, self.flags[idx], self.ids[idx]


def safe_collate_test_csv(batch):
    As, Bs, Ms, flags, ids = zip(*batch)
    As = torch.stack(As)
    Bs = torch.stack(Bs)
    Ms = torch.stack(Ms)
    flags = torch.tensor(flags, dtype=torch.bool)
    return As, Bs, Ms, flags, list(ids)


def get_test_csv_loader(csv_path,
                        batchsize=4,
                        trainsize=256,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True):
    ds = TestCSVChangeDataset(csv_path, trainsize)
    return data.DataLoader(ds,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=True,
                           prefetch_factor=2,
                           collate_fn=safe_collate_test_csv)
