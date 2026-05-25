import asyncio
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse 
from transformers import AutoTokenizer
from utils import Config, TransformerUNet 
import os 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
print(f"Loading model on {DEVICE}...")

cfg = Config()
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
cfg.pad_token_id = tokenizer.pad_token_id

model = TransformerUNet(cfg).to(DEVICE)
model_path = "models/diffusion_slm_checkpoint_loss_updated.pt" 

model = TransformerUNet(cfg).to(DEVICE)
model_path = "models/diffusion_slm_checkpoint_loss_updated.pt" 

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    actual_weights = checkpoint["model_state"]
    model.load_state_dict(actual_weights, strict=False)
    print(f"Successfully loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
else:
    print("Checkpoint not found, using random weights!")

model.eval()

async def generate_diffusion_stream(model, tokenizer, cfg, total_display_steps=40):
    xt = torch.full((1, cfg.seq_len), cfg.mask_token_id, dtype=torch.long, device=DEVICE)
    
    step_size = max(1, cfg.T // total_display_steps)
    
    for t_val in reversed(range(1, cfg.T + 1, step_size)):
        t = torch.tensor([t_val], device=DEVICE)
        
        with torch.no_grad():
            logits = model(xt, t)
            pred_x0 = logits.argmax(dim=-1) 
        
        mask_prob = t_val / cfg.T 
        rand_vals = torch.rand((1, cfg.seq_len), device=DEVICE)
        
        mask = rand_vals < mask_prob
        
        xt = pred_x0.clone()
        xt[mask] = cfg.mask_token_id
        
        current_text = tokenizer.decode(xt[0]).replace(tokenizer.eos_token, "█")
        
        yield current_text, t_val
        await asyncio.sleep(0.05) 

    with torch.no_grad():
        final_logits = model(xt, torch.tensor([1], device=DEVICE))
        final_x0 = final_logits.argmax(dim=-1)
    
    final_text = tokenizer.decode(final_x0[0], skip_special_tokens=True)
    yield final_text, 0

@app.get("/")
async def get_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "index.html")
    
    if not os.path.exists(html_path):
        return {"error": f"index.html not found at {html_path}. Please create it!"}
        
    return FileResponse(html_path)


@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            use_diffusion = data.get("use_diffusion", True)

            if use_diffusion:
                step_counter = 0
                total_steps = 40
                
                async for current_text, t_val in generate_diffusion_stream(model, tokenizer, cfg, total_steps):
                    status = "diffusing" if t_val > 0 else "complete"
                    
                    await websocket.send_json({
                        "step": step_counter,
                        "text": current_text,
                        "status": status
                    })
                    step_counter += 1
            else:
                
                xt = torch.full((1, cfg.seq_len), cfg.mask_token_id, dtype=torch.long, device=DEVICE)
                step_size = max(1, cfg.T // 40)
                
                for t_val in reversed(range(1, cfg.T + 1, step_size)):
                    t = torch.tensor([t_val], device=DEVICE)
                    
                    with torch.no_grad():
                        logits = model(xt, t)
                        pred_x0 = logits.argmax(dim=-1) 
                    
                    mask_prob = t_val / cfg.T 
                    rand_vals = torch.rand((1, cfg.seq_len), device=DEVICE)
                    mask = rand_vals < mask_prob
                    
                    xt = pred_x0.clone()
                    xt[mask] = cfg.mask_token_id
                
                with torch.no_grad():
                    final_logits = model(xt, torch.tensor([1], device=DEVICE))
                    final_x0 = final_logits.argmax(dim=-1)
                    
                text = tokenizer.decode(final_x0[0], skip_special_tokens=True)
                
                await websocket.send_json({
                    "step": 1,
                    "text": text,
                    "status": "complete"
                })

    except WebSocketDisconnect:
        print("Client disconnected")