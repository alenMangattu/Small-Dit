

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import math
import json
from tqdm import tqdm
import random
import requests
from io import BytesIO
from itertools import islice
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

from model import DiT, DiTConfig

def download_image(url: str) -> Optional[Image.Image]:
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception:
        pass
    return None

class CustomImageDataset(Dataset):
    def __init__(self, dataset_name="conceptual_captions", num_samples=1000, image_size=256, num_classes=10):
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])
        
        self.data = self._load_and_filter_data(dataset_name, num_samples)
        if not self.data:
            raise RuntimeError(f"Could not load any valid data from '{dataset_name}'. Please check the dataset or your connection.")

    def _load_and_filter_data(self, dataset_name, num_samples_to_find):
        print(f"Loading dataset '{dataset_name}' in streaming mode...")
        try:
            dataset_stream = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load dataset '{dataset_name}': {e}")
            return []

        print(f"Filtering for {num_samples_to_find} valid image samples...")
        filtered_images = []
        
        max_to_check = num_samples_to_find * 10
        
        pbar = tqdm(islice(dataset_stream, max_to_check), total=num_samples_to_find, desc="Filtering samples")

        for sample in pbar:
            if len(filtered_images) >= num_samples_to_find:
                break

            url = sample.get('image_url') or sample.get('url') # Support different key names
            caption = sample.get('caption') or sample.get('text')

            if url and isinstance(url, str) and caption and isinstance(caption, str):
                img = download_image(url)
                if img:
                    filtered_images.append(img)
                    pbar.update(1)
        
        pbar.close()
        if len(filtered_images) < num_samples_to_find:
            print(f"Warning: Could only find {len(filtered_images)} valid images out of the requested {num_samples_to_find}.")
        
        print(f"Successfully loaded and filtered {len(filtered_images)} images.")
        return filtered_images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        return self.transform(img), label

def get_beta_schedule(num_diffusion_timesteps):
    betas = torch.linspace(0.0001, 0.02, num_diffusion_timesteps)
    return betas

def q_sample(x_start, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]
    
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

@torch.no_grad()
def p_sample_loop(model, shape, num_timesteps, alphas, betas, alphas_cumprod, device, class_label=0):
    img = torch.randn(shape, device=device)
    
    labels = torch.tensor([class_label] * shape[0], device=device)

    for i in tqdm(reversed(range(num_timesteps)), desc="Sampling", total=num_timesteps, leave=False):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        model_to_run = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
        predicted_noise = model_to_run(img, t, labels)
        
        alpha_t = alphas[t][:, None, None, None]
        alpha_cumprod_t = alphas_cumprod[t][:, None, None, None]
        beta_t = betas[t][:, None, None, None]
        
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)
            
        img = 1 / torch.sqrt(alpha_t) * (img - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        
    return img

def show_sample_image(tensor, title=""):
    tensor = (tensor + 1) / 2
    tensor.clamp_(0, 1)
    
    img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

@torch.no_grad()
def validate(model, dataloader, num_diffusion_timesteps, alphas_cumprod, criterion, device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        t = torch.randint(0, num_diffusion_timesteps, (images.shape[0],), device=device).long()
        noise = torch.randn_like(images)
        noisy_images = q_sample(images, t, alphas_cumprod, noise)
        
        predicted_noise = model(noisy_images, t, labels)
        loss = criterion(predicted_noise, noise)
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    model.train()
    return total_loss / len(dataloader)

def main():
    config = DiTConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        n_embd=512,
        n_head=8,
        n_layer=6,
        dropout=0.1
           
    )
    
    epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    num_classes = 10
    num_diffusion_timesteps = 1000
    validation_split = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading dataset from Hugging Face...")
    dataset = CustomImageDataset(
        num_samples=5000, 
        image_size=config.image_size, 
        num_classes=num_classes
    )
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Initializing DiT model on {device}...")
    model = DiT(config, num_classes=num_classes)
    
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
        
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    betas = get_beta_schedule(num_diffusion_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    print("Starting training...")
    step = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            t = torch.randint(0, num_diffusion_timesteps, (images.shape[0],), device=device).long()
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, t, alphas_cumprod, noise)
            
            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(noisy_images, t, labels)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(train_loss=loss.item())
            step += 1

        val_loss = validate(model, val_dataloader, num_diffusion_timesteps, alphas_cumprod, criterion, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1}: Generating sample image...")
        model.eval()
        sample_shape = (1, config.in_channels, config.image_size, config.image_size)
        sample = p_sample_loop(model, sample_shape, num_diffusion_timesteps, alphas, betas, alphas_cumprod, device, class_label=2)
        
        title = f"Sample at Epoch {epoch+1} (Class 2)"
        show_sample_image(sample[0].cpu(), title=title)

    print("Training finished.")

if __name__ == "__main__":
    main()
    
