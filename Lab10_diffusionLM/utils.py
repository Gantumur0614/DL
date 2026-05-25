# 1 imports 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset 

import math 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm 
from einops import rearrange, repeat
from transformers import AutoTokenizer 
from datasets import load_dataset
import warnings 

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F


# 2 hyperparamter configs 
class Config: 
    tokenizer_name = "gpt2"
    vocab_size = 50257
    mask_token_id = 50256 

    seq_len = 128
    d_model = 256 
    n_heads = 8 
    d_ff = 1024 
    dropout = 0.1

    encoder_layers = [2, 2, 2]
    bottleneck_layers = 4 
    decoder_layers = [2, 2, 2]

    T = 1000
    batch_size = 8 
    eval_batch = 4
    lr = 2e-4 
    n_epochs = 30 
    max_steps = 30_000
    grad_clip = 1.0 
    warmup_steps = 500
    n_steps_inf = 50 


#3 Cosine mask scheduler (discrete diffusion)
def cosine_mask_scheduler(T: int, s: float=0.008):
    steps = T + 1 
    t = torch.linspace(0, T, steps)

    alphas_bar = 1 - torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar.max() 
    return alphas_bar 



#4 Forward Diffusion Process

def q_sample(x0: torch.Tensor, t: torch.Tensor, alphas_bar: torch.Tensor, mask_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, L = x0.shape 

    mask_prob = alphas_bar[t]
    mask_prob = mask_prob.unsqueeze(1).expand(B, L)

    mask = torch.bernoulli(mask_prob).bool() 

    xt = x0.clone()
    xt[mask] = mask_token_id
    return xt, mask 


# 5 TransformerUNet architecture
## 5.1 Sinusoidal time embedding 

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model 
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0) 
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

## 5.2 Transformer Block 1D sequence
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.ada_ln_proj = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.ada_ln_proj(time_emb) 
        scale, shift = scale_shift.chunk(2, dim=-1)

        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        h = self.norm1(x) * (1 + scale) + shift 
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out 

        x = x + self.ff(self.norm2(x))
        return x 
    
## 5.3 Unet Scale Modules 

class UNetScale(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)
        ])
    
    def forward(self, x, time_emb):
        for blk in self.blocks:
            x = blk(x, time_emb)
        return x 
    

class Downsample1D(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        x = rearrange(x, "b l c -> b c l")
        x = self.conv(x)
        x = rearrange(x, "b c l -> b l c")
        return self.norm(x)
    

class Upsample1D(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_out)
    
    def forward(self, x):
        x = rearrange(x, "b l c -> b c l ")
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = rearrange(x, "b c l -> b l c")
        return self.norm(x)
    
class SkipFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x_up, x_skip):
        gate = torch.sigmoid(self.alpha)
        fused = gate * x_skip + (1 - gate) * x_up 

        return self.proj(torch.cat([x_up, fused], dim=-1))

##5.4 TransformerUnet

class TransformerUNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        V = cfg.vocab_size
        D = cfg.d_model 
        H = cfg.n_heads 
        F = cfg.d_ff 

        drop = cfg.dropout 
        n_scales = len(cfg.encoder_layers)
        self.self_cond_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.self_cond_gate = nn.Parameter(torch.zeros(1))

        self.token_embed = nn.Embedding(V, D)
        self.pos_embed = nn.Embedding(cfg.seq_len, D)
        self.time_embed = SinusoidalTimeEmbedding(D)
        self.embed_drop = nn.Dropout(drop)

        self.enc_scales = nn.ModuleList() 
        self.downsamples = nn.ModuleList() 

        for i, n_blk in enumerate(cfg.encoder_layers):
            self.enc_scales.append(UNetScale(n_blk, D, H, F, drop))
            if i < n_scales -1:
                self.downsamples.append(Downsample1D(D, D))
        
        self.bottleneck = UNetScale(cfg.bottleneck_layers, D, H, F, drop)

        self.dec_scales = nn.ModuleList()
        self.upsamples = nn.ModuleList() 
        self.skip_fusions = nn.ModuleList() 
        for i, n_blk in enumerate(cfg.decoder_layers):
            if i < n_scales - 1:
                self.upsamples.append(Upsample1D(D, D))
            self.skip_fusions.append(SkipFusion(D))
            self.dec_scales.append(UNetScale(n_blk, D, H, F, drop))

        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, V)

        self._init_weights() 

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, x_prev_pred=None) -> torch.Tensor:
        B, L = xt.shape 
        n_scales = len(self.enc_scales)
        positions = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(xt) + self.pos_embed(positions)
        if x_prev_pred is not None: 
            gate = torch.sigmoid(self.self_cond_gate)
            x = x + gate * self.self_cond_embed(x_prev_pred)
        x = self.embed_drop(x)

        time_emb = self.time_embed(t)

        skips = []
        for i, enc in enumerate(self.enc_scales):
            x = enc(x, time_emb)
            skips.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.bottleneck(x, time_emb)

        up_idx = 0 
        for i, (dec, fuse) in enumerate(zip(self.dec_scales, self.skip_fusions)):
            skip_idx = n_scales - 1 - i 
            if i > 0:
                x = self.upsamples[up_idx](x)
                up_idx = up_idx + 1
            x = fuse(x, skips[skip_idx])
            x = dec(x, time_emb)
        
        x = self.out_norm(x)
        logits = self.out_proj(x)
        return logits 
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        return total 


#6 Training Objectives

def sample_timesteps(B, T, device, low_t_bias=0.5):
    if torch.rand(1) < low_t_bias:
        t = torch.randint(1, T // 4 + 1, (B,), device=device)
    else:
        t = torch.randint(1, T + 1, (B,), device=device)
    return t 

# 7 For only text dataset
class TextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len 
        n = (len(token_ids) // seq_len) * seq_len 
        self.data = token_ids[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 7 Dataset - MN -> EN     

class TextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        n = (len(token_ids) // seq_len) * seq_len
        self.data = token_ids[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#8 Optimizer, LR schedul & Training Loop

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps) 
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5, (1.0 + math.cos(math.pi * progress))) 
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



# 9 diffusion Loss

def diffusion_loss(model, x0, alphas_bar, cfg):
    B, L = x0.shape
    device = x0.device

    T = len(alphas_bar) - 1 
    t = torch.randint(1, T + 1, (B,), device=device)

    mask_probs = alphas_bar[t].view(B, 1)

    rand_vals = torch.rand((B, L), device=device)
    
    pad_id = getattr(cfg, 'pad_token_id', 50256) 
    is_not_pad = (x0 != pad_id)
    
    mask = (rand_vals < mask_probs) & is_not_pad

    x_t = x0.clone()
    x_t[mask] = cfg.mask_token_id

    logits = model(x_t, t) 

    logits_flat = logits.view(-1, logits.size(-1))
    x0_flat = x0.view(-1)

    loss_ce = F.cross_entropy(logits_flat, x0_flat, reduction='none')
    loss_ce = loss_ce.view(B, L)
    
    masked_loss = (loss_ce * mask).sum() / mask.sum().clamp(min=1e-5)

    return masked_loss, mask.sum()




















