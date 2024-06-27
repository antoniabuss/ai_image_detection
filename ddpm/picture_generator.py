import os
import torch
from diffusers import DDPMPipeline
from PIL import Image
import uuid  # Import uuid module for generating random strings
from torch.cuda.amp import autocast, GradScaler

def generate_and_save_images(num_images=100, num_inference_steps=1000, seed=42, use_cuda=True, output_dir="Pictures"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model_id = "google/ddpm-celebahq-256"
    pipeline = DDPMPipeline.from_pretrained(model_id)
    
    # Use GPU if available and desired
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    pipeline.to(device)

    # Generate and save images
    for i in range(num_images):
        generator = torch.manual_seed(seed + i)  # Adjust seed for each image

        # Use autocast for mixed precision
        scaler = GradScaler(enabled=use_cuda)
        with autocast(enabled=use_cuda):
            image = pipeline(generator=generator, num_inference_steps=num_inference_steps).images[0]

        # Generate a random string for the image filename
        unique_filename = uuid.uuid4().hex + ".png"
        image_path = os.path.join(output_dir, unique_filename)
        
        # Save the generated image
        image.save(image_path)
        print(f"Saved image {i+1}/{num_images} at: {image_path}")

        # Optionally display the image using matplotlib
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()

if __name__ == "__main__":
    # Parameters
    num_images = 1000
    num_inference_steps = 100
    seed =  0		# acts as the startingpoint of the model
    use_cuda = True
    output_dir = "C:\\Users\\wolf-\\anaconda3\\envs\\celeba-diffusion\\Scripts\\Pictures"  # Adjust as per your environment

    # Generate and save images
    generate_and_save_images(num_images=num_images, num_inference_steps=num_inference_steps, seed=seed, use_cuda=use_cuda, output_dir=output_dir)

