import argparse
import os
import torch
from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the directory containing the original training images.')
    parser.add_argument('--fractal_dir', type=str, required=True, help='Path to the directory containing the fractal images.')
    parser.add_argument('--prompts', type=str, required=True, help='Comma-separated list of prompts for image generation.')
    parser.add_argument('--multi_domain', type=bool, required=True, help='Specify whether the dataset has multiple domains.')
    parser.add_argument('--num_images', type=int, required=True, help="Determine the number of processing images for one time.")
    return parser.parse_args()

def augment_domain(domain, domain_path, args, prompts):
    
    # Initialize the model handler, for pix2pix
    # model_id = "timbrooks/instruct-pix2pix"
    # model_initialization = ModelHandler(model_id=model_id, device='cuda')

    # Initialize the model pipeline for ControlNet
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", use_safetensors=True)

    # img2img pipeline
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        # "SimianLuo/LCM_Dreamshaper_v7", # utilize LCM
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker = None,
        requires_safety_checker = False
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()


    # Load the original dataset
    train_dataset = datasets.ImageFolder(root=domain_path)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    negative_prompt = '''lowres, bad anatomy, bad hands, text,error, missing fngers,extra digt ,fewer digits,cropped, wort quality ,low quality,normal quality, jpeg artifacts,signature,watermark, username, blurry, bad feet,'''
    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        domain=domain,
        pipe=pipe,
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=args.num_images,
        guidance_scale=4,
        idx_to_class = idx_to_class,
        prompts = prompts,
        negative_prompt=negative_prompt
    )


def main():
    args = parse_arguments()
    prompts = args.prompts.split(',')  # This will give you a list of prompts

    if (args.multi_domain):
        domain_lst = os.listdir(args.train_dir)
        for domain in domain_lst:
            domain_path = os.path.join(args.train_dir, domain)
            if os.path.isdir(domain_path):
                print(f"augmenting domain: {domain}")
                augment_domain(domain, domain_path, args, prompts)
    
    else:
        domain_path = args.train_dir
        augment_domain('result', domain_path, args, prompts)

if __name__ == '__main__':
    main()


