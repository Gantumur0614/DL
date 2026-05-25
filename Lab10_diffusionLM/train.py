import torch 
from transformers import AutoTokenizer 
from tqdm.auto import tqdm 
from datasets import load_dataset 
from utils import * 

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config() 
alphas_bar = cosine_mask_scheduler(cfg.T).to(device)

model = TransformerUNet(cfg).to(device)

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
cfg.pad_token_id = tokenizer.pad_token_id 

raw_train_set = load_dataset("Ganaa0614/mongolian-text-dataset", split="train")
raw_val_set = load_dataset("Ganaa0614/mongolian-text-dataset", split="test")

train_set = TextDataset(raw_train_set, tokenizer, cfg.seq_len)
val_set = TextDataset(raw_val_set, tokenizer, cfg.seq_len)

train_loader = DataLoader(
    train_set,
    cfg.batch_size,
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    cfg.eval_batch,
    num_workers=4,
    pin_memory=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    cfg.lr,
    betas=[0.9, 0.999],
    weight_decay=0.01
)

total_steps = min(cfg.max_steps, cfg.n_epochs * len(train_set))
scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

train_losses = []
global_step  = 0


model.train()

for epoch in range(cfg.n_epochs):
    epoch_loss = 0.0
    n_batches  = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.n_epochs}', leave=True)

    for batch in pbar:
        if global_step >= cfg.max_steps:
            break

        x0 = batch.to(device)  

        optimizer.zero_grad()
        loss, n_masked = diffusion_loss(model, x0, alphas_bar, cfg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        epoch_loss += loss_val
        n_batches  += 1
        global_step += 1

        train_losses.append(loss_val)
        lr_now = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f'{loss_val:.4f}', lr=f'{lr_now:.2e}',
                         masked=int(n_masked))

    avg = epoch_loss / max(n_batches, 1)
    print(f'Epoch {epoch+1} avg loss: {avg:.4f} | steps: {global_step}')

    if global_step >= cfg.max_steps:
        print('Reached max_steps, stopping.')
        break