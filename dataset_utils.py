"""
Dataset Utility Functions for Image-Text Datasets
Clean implementation with no bloatware
"""

from datasets import load_dataset
from itertools import islice
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def load_dataset_streaming(dataset_name: str, num_samples: int = 5000) -> List[Dict]:
    """Load dataset in streaming mode and collect specified number of samples."""
    try:
        print(f"Loading {dataset_name} in streaming mode...")
        dataset_stream = load_dataset(dataset_name, split="train", streaming=True)
        
        samples = []
        for idx, sample in enumerate(islice(dataset_stream, num_samples)):
            samples.append(sample)
            if (idx + 1) % 500 == 0:
                print(f"  Collected {idx + 1} samples...")
        
        print(f"✅ Successfully collected {len(samples)} samples")
        return samples
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        # Fallback to alternative dataset
        print("Trying alternative dataset...")
        return load_dataset_streaming("conceptual_captions", num_samples=1000)


def download_image(url: str) -> Optional[Image.Image]:
    """Download image from URL and return PIL Image object."""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
    return None


def get_sample(samples: List[Dict], index: int) -> Dict[str, Any]:
    """
    Get data at specific index with proper handling.
    
    Returns dict with keys:
    - index: sample index
    - prompt: instruction/prompt text
    - caption: target caption/description
    - input_image: source PIL Image (if available)
    - output_image: target PIL Image
    """
    if 0 <= index < len(samples):
        sample = samples[index]
        
        # Standardize the output format
        output = {
            'index': index,
            'prompt': sample.get('prompt', sample.get('caption', 'No prompt')),
            'caption': sample.get('target_caption', sample.get('caption', 'No caption')),
            'input_image': None,
            'output_image': None
        }
        
        # Handle different dataset formats
        if 'target_image' in sample:
            # MetaQuery format - actual image objects
            output['output_image'] = sample['target_image']
            if 'source_images' in sample and len(sample['source_images']) > 0:
                output['input_image'] = sample['source_images'][0]
        
        elif 'image_url' in sample or 'image' in sample:
            # URL-based format - download images
            url = sample.get('image_url') or sample.get('image')
            output['output_image'] = download_image(url)
        
        return output
    else:
        raise IndexError(f"Index {index} out of range. Dataset has {len(samples)} samples.")


def get_batch(samples: List[Dict], indices: List[int]) -> Dict[str, List]:
    """
    Get a batch of samples with prompts, input images, and output images.
    
    Returns dict with keys:
    - prompts: List of prompt strings
    - input_images: List of PIL Images (or None)
    - output_images: List of PIL Images (or None)
    - captions: List of caption strings
    """
    batch = {
        'prompts': [],
        'input_images': [],
        'output_images': [],
        'captions': []
    }
    
    for idx in indices:
        try:
            sample = get_sample(samples, idx)
            batch['prompts'].append(sample['prompt'])
            batch['input_images'].append(sample['input_image'])
            batch['output_images'].append(sample['output_image'])
            batch['captions'].append(sample['caption'])
        except IndexError as e:
            print(f"Skipping index {idx}: {e}")
    
    return batch


# Quick usage example:
if __name__ == "__main__":
    # Load first 100 samples
    samples = load_dataset_streaming("conceptual_captions", num_samples=100)
    
    # Get single sample
    sample = get_sample(samples, 0)
    print(f"Sample 0: {sample['prompt']}")
    
    # Get batch
    batch = get_batch(samples, [0, 1, 2, 3, 4])
    print(f"Batch size: {len(batch['prompts'])}") 