import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import requests
from tqdm import tqdm
import random
import time
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

# Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# 简单LLM
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.fc(x)
        return x

# 数据加载器
class TextDataset(Dataset):
    def __init__(self, file_path, vocab, seq_length=64):
        self.vocab = vocab
        self.seq_length = seq_length
        
        # 预先将所有序列转换为张量并缓存
        self.sequences = []
        
        print("Loading and preprocessing data...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in tqdm(lines, desc="Processing sequences"):
            tokens = line.split()
            if len(tokens) > self.seq_length:
                for i in range(0, len(tokens) - self.seq_length, self.seq_length // 2):
                    seq = tokens[i:i + self.seq_length]
                    tensor = torch.LongTensor(
                        [self.vocab.get(token.lower(), self.vocab['<unk>']) for token in seq]
                    )
                    self.sequences.append(tensor)
            else:
                tensor = torch.LongTensor(
                    [self.vocab.get(token.lower(), self.vocab['<unk>']) for token in tokens] +
                    [self.vocab['<pad>']] * (self.seq_length - len(tokens))
                )
                self.sequences.append(tensor)
        
        print(f"Total sequences loaded: {len(self.sequences)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def download_wikitext(url, save_path):
    """下载数据集"""
    if os.path.exists(save_path):
        print(f"Dataset already exists at {save_path}, skipping download...")
        return
        
    print(f"Downloading dataset from {url}")
    print(f"This may take a few minutes depending on your internet connection...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"Download completed! File saved to {save_path}")

def build_vocab(file_path, max_vocab_size=10000):
    """构建词表"""
    print(f"\nBuilding vocabulary from {file_path}")
    print(f"Maximum vocabulary size: {max_vocab_size}")
    
    word_freq = {}
    total_words = 0
    unique_words = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                word = word.lower()
                total_words += 1
                if word not in word_freq:
                    unique_words += 1
                word_freq[word] = word_freq.get(word, 0) + 1
    
    print(f"Total words in dataset: {total_words:,}")
    print(f"Unique words before filtering: {unique_words:,}")
    
    # 按频率排序并选择最常见的词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    selected_words = sorted_words[:max_vocab_size-4]  # 留出空间给特殊token
    
    # 构建词表
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3,
    }
    for word, freq in selected_words:
        vocab[word] = len(vocab)
    
    # 打印词表统计信息
    coverage = sum(freq for _, freq in selected_words) / total_words * 100
    print(f"Vocabulary size after filtering: {len(vocab):,}")
    print(f"Vocabulary coverage: {coverage:.2f}%")
    
    return vocab

def train(model, data_loader, epochs=10, lr=0.001):
    # 检测是否可以使用MPS（Apple Silicon）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nTraining parameters:")
    print(f"- Device: {device}")
    print(f"- Learning rate: {lr}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {data_loader.batch_size}")
    print(f"- Steps per epoch: {len(data_loader)}")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, batch in enumerate(data_loader):
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                
                seq_len = inputs.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
                
                outputs = model(inputs, mask)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                current_lr = optimizer.param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                pbar.update(1)
        
        scheduler.step()
        epoch_loss = total_loss / len(data_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"- Loss: {epoch_loss:.4f}")
        print(f"- Time: {epoch_time:.2f}s")
        print(f"- Samples/second: {len(data_loader.dataset)/epoch_time:.2f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best loss achieved!")
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pt')
            print("Model checkpoint saved")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Best loss achieved: {best_loss:.4f}")

def generate(model, prompt, vocab, max_len=50, temperature=1.0):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.eval()
    tokens = [vocab.get(token.lower(), vocab['<unk>']) for token in prompt.split()]
    
    print(f"\nGenerating text with parameters:")
    print(f"- Prompt: '{prompt}'")
    print(f"- Max length: {max_len}")
    print(f"- Temperature: {temperature}")
    
    generated_tokens = []
    start_time = time.time()
    
    for i in range(max_len):
        inputs = torch.LongTensor([tokens]).to(device)
        seq_len = inputs.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
        
        with torch.no_grad():
            outputs = model(inputs, mask)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1).item()
            
            if next_token == vocab.get('<eos>', 0):
                break
                
            tokens.append(next_token)
            generated_tokens.append(next_token)
    
    generation_time = time.time() - start_time
    tokens_per_second = len(generated_tokens) / generation_time
    
    result = ' '.join([list(vocab.keys())[list(vocab.values()).index(t)] for t in tokens])
    
    print(f"\nGeneration stats:")
    print(f"- Tokens generated: {len(generated_tokens)}")
    print(f"- Generation time: {generation_time:.2f}s")
    print(f"- Tokens/second: {tokens_per_second:.2f}")
    
    return result

# 主程序
if __name__ == "__main__":
    # 设置多进程数据加载
    mp.set_start_method('spawn')
    
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
    print(f"Number of CPU cores: {os.cpu_count()}")
    
    # 下载WikiText-2数据集
    train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    train_path = "wikitext2_train.txt"
    download_wikitext(train_url, train_path)

    # 构建词表
    vocab = build_vocab(train_path)
    print(f"Vocabulary size: {len(vocab):,}")

    # 加载数据
    print("\nPreparing dataset...")
    dataset = TextDataset(train_path, vocab, seq_length=128)
    # 使用多进程数据加载
    data_loader = DataLoader(
        dataset, 
        batch_size=128,  # 增加批次大小
        shuffle=True,
        num_workers=min(4, os.cpu_count()),  # 使用多进程加载
        pin_memory=True  # 提高数据传输效率
    )
    print(f"Dataset size: {len(dataset):,} sequences")

    # 初始化更大的模型
    print("\nInitializing model...")
    model = SimpleLLM(
        vocab_size=len(vocab),
        d_model=256,
        n_heads=8,
        n_layers=6
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture:")
    print(f"- Vocabulary size: {len(vocab):,}")
    print(f"- Model dimension: 256")
    print(f"- Attention heads: 8")
    print(f"- Transformer layers: 6")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")

    # 训练模型
    print("\nStarting training...")
    train(model, data_loader, epochs=5, lr=0.0001)

    # 生成文本
    print("\nGenerating sample texts...")
    prompts = [
        "the king",
        "in ancient times",
        "scientists discovered",
        "the purpose of"
    ]
    
    for prompt in prompts:
        generated_text = generate(model, prompt, vocab, max_len=100, temperature=0.8)
        print(f"\nFinal generated text for '{prompt}':")
        print(f"{generated_text}")

    print(f"\nProcess completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")