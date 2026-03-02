"""
공통 유틸리티 함수
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path


def save_model(model, save_path, model_name="model"):
    """모델 저장"""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ 모델 저장 완료: {model_path}")
    return model_path


def load_model(model, model_path):
    """모델 로드"""
    model.load_state_dict(torch.load(model_path))
    print(f"✅ 모델 로드 완료: {model_path}")
    return model


def plot_training_history(history, save_path=None):
    """학습 히스토리 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 학습 곡선 저장: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion Matrix 저장: {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """분류 리포트 출력"""
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 CPU 사용")
    return device


class EarlyStopping:
    """Early Stopping 구현"""
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'max'
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop
