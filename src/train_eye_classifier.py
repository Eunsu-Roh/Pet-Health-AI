"""
안구 질환 분류 모델 학습
EfficientNet-B2 사용
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from tqdm import tqdm
from pathlib import Path
import sys

# 상위 디렉토리 경로 추가
sys.path.append(str(Path(__file__).parent))

from dataset import get_dataloaders
from utils import (
    get_device, save_model, plot_training_history,
    plot_confusion_matrix, print_classification_report, EarlyStopping
)


def create_model(num_classes):
    """EfficientNet-B2 모델 생성"""
    model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    
    # 마지막 레이어 교체
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """1 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def main():
    # 설정
    DATA_DIR = "data/dog_eye_classification"  # YOLO에서 변환된 분류 데이터
    SAVE_DIR = "models/eye_classifier"
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    
    print("=" * 60)
    print("🐕 안구 질환 분류 모델 학습 시작")
    print("=" * 60)
    
    # 디바이스 설정
    device = get_device()
    
    # 데이터 로더
    print("\n📦 데이터 로딩 중...")
    train_loader, val_loader, class_to_idx, idx_to_class = get_dataloaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )
    
    num_classes = len(class_to_idx)
    print(f"📊 클래스 수: {num_classes}")
    print(f"   클래스: {list(idx_to_class.values())}\n")
    
    # 모델 생성
    print("🔨 모델 생성 중...")
    model = create_model(num_classes).to(device)
    print(f"✅ EfficientNet-B2 모델 준비 완료\n")
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
    
    # TensorBoard
    writer = SummaryWriter(f'runs/eye_classifier')
    
    # 학습 히스토리
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0
    
    # 학습 루프
    print("🚀 학습 시작!\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # TensorBoard 로깅
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        # 결과 출력
        print(f"\n📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%\n")
        
        # 최고 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, SAVE_DIR, "best_model")
            print(f"🌟 최고 성능 모델 저장! Val Acc: {best_acc*100:.2f}%\n")
        
        # Early Stopping
        if early_stopping(val_acc):
            print(f"⚠️  Early Stopping at epoch {epoch+1}")
            break
    
    writer.close()
    
    # 최종 모델 저장
    save_model(model, SAVE_DIR, "final_model")
    
    # 학습 곡선 시각화
    print("\n📈 학습 곡선 생성 중...")
    plot_training_history(history, f"{SAVE_DIR}/training_history.png")
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix 생성 중...")
    plot_confusion_matrix(
        val_labels, val_preds, 
        list(idx_to_class.values()),
        f"{SAVE_DIR}/confusion_matrix.png"
    )
    
    # Classification Report
    print_classification_report(val_labels, val_preds, list(idx_to_class.values()))
    
    print("\n" + "=" * 60)
    print(f"🎉 학습 완료! 최고 검증 정확도: {best_acc*100:.2f}%")
    print(f"📁 모델 저장 위치: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
