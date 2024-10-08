import argparse
import os
from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the directory containing the original training images.')
    parser.add_argument('--fractal_dir', type=str, required=True, help='Path to the directory containing the fractal images.')
    parser.add_argument('--prompts', type=str, required=True, help='Comma-separated list of prompts for image generation.')
    parser.add_argument('--multi_domain', type=bool, required=True, help='Specify whether the dataset has multiple domains.')
    parser.add_argument('--num_images', type=int, required=True, help="Determine the number of processing images for one time.")
    return parser.parse_args()

def augment_domain(domain, domain_path, args, prompts):
    
    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    # Load the original dataset
    train_dataset = datasets.ImageFolder(root=domain_path)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        domain=domain,
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=args.num_images,
        guidance_scale=4,
        idx_to_class = idx_to_class,
        prompts=prompts,
        model_handler=model_initialization
    )

    # for idx, (image, label) in enumerate(augmented_train_dataset):
    #     image.save(f'augmented_images/{idx}.png')
    #     pass

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


