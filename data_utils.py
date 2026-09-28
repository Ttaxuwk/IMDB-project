import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import re
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import re
def tokenize(text):# 对文本进行分词
    text=text.lower()
    text=re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

#构建词表
class Vocab:
    def __init__(self,texts,max_size=10000,min_freq=2):
        counter=Counter()
        for text in texts:
            counter.update(tokenize(text))# 对每个文本进行分词并更新词频统计
        #特殊token,空位、未知词处理
        self.pad_idx =0
        self.unk_idx =1

        #按频率从高到低取词，少于2的直接丢掉，没有学习意义
        self.word2idx={'<pad>':0,'<unk>':1}
        idx=2
        for word,freq in counter.most_common(max_size-2):
            if freq<min_freq:
                break
            self.word2idx[word]=idx
            idx+=1
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.vocab_size=len(self.word2idx)

    def encode(self, text, max_len=256):
        tokens=tokenize(text)
        ids =[self.word2idx.get(t,self.unk_idx) for t in tokens]

        #padding/截断
        if len(ids)<max_len:
            ids=ids+[self.pad_idx]*(max_len-len(ids))
        else:
            ids=ids[:max_len]
        return ids

#数据集类
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts=texts
        self.labels=labels
        self.vocab=vocab
        self.max_len=max_len

    def __len__(self):
        return len(self.texts)

    #我们需要使用之前定义的类和函数，把文本转化为索引，然后再做张量化
    def __getitem__(self, idx):
        ids=self.vocab.encode(self.texts[idx],self.max_len)
        return torch.tensor(ids,dtype=torch.long),torch.tensor(self.labels[idx],dtype=torch.long)

#加载IMDB
def load_imdb_data():
    print("加载数据集...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # 从训练集划分验证集
    split_point = int(len(train_texts) * 0.9)
    val_texts = train_texts[split_point:]
    val_labels = train_labels[split_point:]
    train_texts = train_texts[:split_point]
    train_labels = train_labels[:split_point]

    print(f"训练集: {len(train_texts)}")
    print(f"验证集: {len(val_texts)}")
    print(f"测试集: {len(test_texts)}")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def get_dataloaders(batch_size=64, max_len=256, max_vocab=10000):
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels=load_imdb_data()

    #用训练集构建词表
    print("构建词表...")
    vocab=Vocab(train_texts,max_size=max_vocab,min_freq=2)
    print(f"词表大小为:{vocab.vocab_size}")

    train_set = IMDBDataset(train_texts, train_labels, vocab, max_len)
    val_set = IMDBDataset(val_texts, val_labels, vocab, max_len)
    test_set = IMDBDataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, vocab