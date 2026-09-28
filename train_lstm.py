import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_dataloaders

#=========== Part 1: 基础设置==========
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# ========== Part 2: 数据 ==========
train_loader, val_loader, test_loader, vocab = get_dataloaders(batch_size=64, max_len=256)

# ========== Part 3: LSTM 模型 ==========
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_classes=2, dropout =0.5, pad_idx=0):
        super().__init__()
        self.pad_idx=pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    #前向传播
    def forward(self,x):
        mask=(x!=self.pad_idx)
        embedded = self.dropout(self.embedding(x))
        output,(h_n, c_n)=self.lstm(embedded)
        mask_float=mask.unsqueeze(-1).float()
        summed = (output * mask_float).sum(dim=1)
        counts = mask_float.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
model = LSTMModel(vocab_size=vocab.vocab_size, embed_dim=128, hidden_dim=128, pad_idx=vocab.pad_idx).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量:{total_params:,}")
dummy = torch.randint(0, vocab.vocab_size, (4, 256)).to(device)
out = model(dummy)
print("输出形状:", out.shape)

# ========== Part 4: 训练与评估函数 ==========
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

# ========== Part5: 主训练循环 ==========
@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

epochs = 20
patience = 5
counter = 0
best_val_loss = float('inf')
best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'models/best_lstm.pth')
    else:
        counter += 1

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
          f"lr {current_lr:.6f}")

    if counter >= patience:
        print(f"\n早停：连续 {patience} 个 epoch 验证损失未改善")
        break

print(f"\n最佳验证准确率: {best_val_acc:.4f}")

# ========== Part6: 测试集评估 ==========
print("\n在测试集上评估最佳模型...")
model.load_state_dict(torch.load('models/best_lstm.pth'))
test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集损失: {test_loss:.4f}")

# ========== Part7: 画训练曲线 ==========
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('LSTM Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/lstm_training_curve.png', dpi=100)
plt.close()
print("已保存训练曲线到 results/lstm_training_curve.png")