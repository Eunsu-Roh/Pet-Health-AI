"""
커스텀 데이터셋 클래스
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class ImageClassificationDataset(Dataset):
    """이미지 분류를 위한 데이터셋"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: 데이터 루트 디렉토리
            split: 'train', 'valid', 'test'
            transform: Albumentations transform
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 클래스 폴더 읽기
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 이미지 경로 및 라벨 수집
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"📊 {split} 데이터셋: {len(self.images)}장")
        print(f"   클래스: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드 (OpenCV for Albumentations)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform 적용
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_transforms(img_size=224, split='train'):
    """데이터 증강 및 전처리"""
    
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def get_dataloaders(root_dir, batch_size=32, img_size=224, num_workers=4):
    """DataLoader 생성"""
    train_dataset = ImageClassificationDataset(
        root_dir, 
        split='train',
        transform=get_transforms(img_size, 'train')
    )
    
    val_dataset = ImageClassificationDataset(
        root_dir,
        split='valid',
        transform=get_transforms(img_size, 'valid')
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx, train_dataset.idx_to_class
