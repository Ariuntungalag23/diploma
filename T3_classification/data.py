"""
================================================================================
DATA — PyTorch Dataset, 16-bit рентген унших, augmentation
================================================================================
"""

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG


# ImageNet статистик (timm моделиуд бүгд үүнийг ашигладаг)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ============================================================================
# 16-BIT ЗУРАГ УНШИХ + WINDOW/LEVEL NORMALIZATION
# ============================================================================

def read_image(path: str, cfg: CFG = None) -> np.ndarray:
    """
    Зураг уншиж 8-bit RGB numpy array болгож буцаана.
    
    GRAZPEDWRI-DX-ийн 16-bit PNG → window/level normalization-аар 8-bit рүү map
    Бусад dataset-ийн 8-bit JPG/PNG → шууд унших
    
    Returns: uint8 numpy array, shape (H, W, 3), RGB
    """
    if cfg is None:
        cfg = CFG()
    
    # cv2.IMREAD_UNCHANGED → 16-bit, 8-bit, RGB, grayscale бүгдийг хадгална
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Зураг уншиж чадсангүй: {path}")
    
    # ---- 16-bit бол window/level хийх ----
    if img.dtype == np.uint16:
        # Доод p_low%, дээд p_high% percentile-ийг тасалж 8-bit рүү дэлгэх
        # Хэрэв зургийн ихэнх нь хар background бол percentile-ыг
        # foreground дээрээс л тооцох нь зөв (background-ийг 0 гэж хасах)
        nonzero = img[img > 0]
        if len(nonzero) > 100:
            p_low, p_high = np.percentile(
                nonzero, [cfg.wl_p_low, cfg.wl_p_high]
            )
        else:
            p_low, p_high = img.min(), img.max()
        
        if p_high <= p_low:
            p_high = p_low + 1
        
        img = np.clip(img, p_low, p_high)
        img = ((img - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    
    # ---- Channel-уудыг тааруулах ----
    if img.ndim == 2:
        # Grayscale → 3-channel RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img


# ============================================================================
# AUGMENTATION (рентген-д тохирсон)
# ============================================================================

def build_transforms(img_size: int, train: bool):
    """
    Albumentations pipeline.
    
    ЧУХАЛ — рентгенд анхаарах зүйлс:
      • VerticalFlip ХИЙХГҮЙ (рентгенд антомийн утга бий, баруун/зүүн чухал)
      • HorizontalFlip OK (зүүн/баруун гар адил)
      • Rotate ±20° хүртэл (хэт олон бол анатоми гажин)
      • CLAHE — рентгенд контраст сайжруулдаг
    """
    if train:
        return A.Compose([
            # Хэмжээ
            A.LongestMaxSize(max_size=int(img_size * 1.15)),
            A.PadIfNeeded(
                min_height=int(img_size * 1.15),
                min_width=int(img_size * 1.15),
                border_mode=cv2.BORDER_CONSTANT, value=0
            ),
            A.RandomCrop(height=img_size, width=img_size),
            
            # Геометр augmentation
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.07, scale_limit=0.15, rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT, p=0.7
            ),
            
            # Контраст / гэрэлтүүлэг
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.7),
            
            # Бүүр / sharpen
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.Sharpen(p=1.0),
            ], p=0.3),
            
            # Random erasing (CoarseDropout)
            A.CoarseDropout(
                max_holes=3,
                max_height=img_size // 10,
                max_width=img_size // 10,
                fill_value=0, p=0.3
            ),
            
            # Normalize + ToTensor
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        # Validation/test: зөвхөн resize + normalize
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT, value=0
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def build_tta_transform(img_size: int, hflip: bool = False):
    """
    TTA-д ашиглах transform — өөр өөр хэмжээ, hflip-тэй
    """
    tfs = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT, value=0
        ),
    ]
    if hflip:
        tfs.append(A.HorizontalFlip(p=1.0))
    tfs.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return A.Compose(tfs)


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class FractureDataset(Dataset):
    """
    Bone fracture classification dataset.
    
    Аргумент:
      df       : pd.DataFrame, баганууд: image_path, label
      transform: albumentations Compose
      cfg      : CFG (16-bit window/level-д ашиглана)
    """
    
    def __init__(self, df: pd.DataFrame, transform=None, cfg: CFG = None):
        if 'image_path' not in df.columns or 'label' not in df.columns:
            raise ValueError("df-д 'image_path', 'label' багана байх ёстой")
        self.paths = df['image_path'].values
        self.labels = df['label'].values.astype(np.float32)
        self.transform = transform
        self.cfg = cfg or CFG()
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            img = read_image(self.paths[idx], self.cfg)
        except Exception as e:
            # Зураг гэмтсэн бол өөр зураг ашиглах (training-ийг таслахгүй)
            print(f"⚠ Зураг уншихад алдаа: {self.paths[idx]} ({e})")
            # Хоосон 0-тэй зураг буцаах
            img = np.zeros((self.cfg.img_size, self.cfg.img_size, 3), dtype=np.uint8)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)
