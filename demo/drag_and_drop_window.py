import os
import sys
import torch
from torchvision.utils import saveimage

#Add the StyleGAN3 directory to the Python path
sys.path.append('stylegan3')

import dnnlib
import legacy

#Path to the pretrained model
network_pkl = 'models/stylegan3/stylegan3-t-ffhqu-256x256.pkl'  # Adjust the path as necessary
device = torch.device('cpu')  # Use CPU

#Load pretrained model
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Load the generator

#Directory to save the generated images
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

#Number of images to generate
num_images = 1001
start_index = 999  # Adjust the start index as needed

truncation_psi = 0.5  # Truncation-Wert einstellen, z.B. 0.5

def generate_images(start_index, num_images, output_dir):
    for i in range(start_index, num_images):
        z = torch.randn([1, G.z_dim], device=device)  # Generate latent code
        c = None  # Class labels (not used in this example)
        img = G(z, c, truncation_psi=truncation_psi)  # Generate image with truncation value
        img = (img.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
        save_image(img, os.path.join(output_dir, f'generated_image{i:04d}.png'))  # Save image
        print(f'Generated {i + 1}/{num_images} images')

#Generate the images
generate_images(start_index, num_images, output_dir)