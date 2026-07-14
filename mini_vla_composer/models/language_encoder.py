"""简单词表 tokenizer 和语言编码器。"""

import re

import torch
from torch import nn


class SimpleTokenizer:
    """面向固定任务词汇的轻量 tokenizer。"""

    def __init__(self, max_len: int = 12) -> None:
        """初始化固定词表。"""
        words = [
            "<pad>",
            "<unk>",
            "move",
            "put",
            "the",
            "red",
            "blue",
            "green",
            "yellow",
            "square",
            "circle",
            "triangle",
            "to",
            "into",
            "target",
            "zone",
            "area",
        ]
        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = words
        self.max_len = max_len

    @property
    def vocab_size(self) -> int:
        """返回词表大小。"""
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """把指令文本编码为固定长度 token id。"""
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        ids = [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens[: self.max_len]]
        ids += [self.stoi["<pad>"]] * (self.max_len - len(ids))
        return ids


class LanguageEncoder(nn.Module):
    """Embedding 加平均池化得到语言特征。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        output_dim: int = 64,
    ) -> None:
        """创建语言编码层。"""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.ReLU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """输入 B x L token，输出 B x output_dim 特征。"""
        emb = self.embedding(tokens)
        mask = (tokens != 0).float().unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(pooled)
