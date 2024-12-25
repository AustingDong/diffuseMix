import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from augment.utils import Utils
from transformers import pipeline


class DiffuseMix(Dataset):
    def __init__(self, domain, pipe, original_dataset, num_images, guidance_scale, fractal_imgs, idx_to_class, prompts, negative_prompt
                #  model_handler
                 ):
        self.domain = domain
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.fractal_imgs = fractal_imgs
        self.prompts = prompts
        self.negative_prompt = negative_prompt

        self.resize_shape = (512, 512) # shape of resized image

        # self.model_handler = model_handler # Original model handler
        self.model_pipe = pipe # Model pipeline
        self.depth_estimator = pipeline("depth-estimation")

        # self.num_augmented_images_per_image = num_images
        self.batch_size=num_images
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.augmented_images = self.generate_augmented_images()

    
    def expand_prompt_description(self, prompt):
        art_method = random.choice(['Oil Painting', 'Watercolor', 'Digital Painting', 'Crayon Painting'])
        art_details = random.choice(['Soft brushstrokes', 'intricate textures', 'subtle atmospheric perspective'])
        prompt_description = {
            "art_painting": f"{art_method}, {art_details}",
            "cartoon": "thick outlines, vivid colors, simple shapes, Disney-like style, playful atmosphere, flat 2D perspective",
            "sketch": "{pencil sketch}, {line art}, simple black-and-white sketch style, minimal shading",
            "photo": "realistic lighting, camera-like details, natural, Detailed skin texture, neutral color grading"
        }
        return prompt_description[prompt]


    def get_canny_image(self, original_img):
        low_threshold = 100
        high_threshold = 200
        original_img_arr = np.array(original_img)
        canny_img = cv2.Canny(original_img_arr, low_threshold, high_threshold)
        canny_img = canny_img[:, :, None]
        canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
        canny_image = Image.fromarray(canny_img)
        return canny_image
        

    def get_depth_map(self, image):
        image = self.depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        depth_map = Image.fromarray(image)
        return depth_map

    def generate_augmented_images(self, method="depth_map"):
        augmented_data = []
        original_img_batch = []
        control_img_batch = []
        img_filename_batch = []
        label_batch = []
        label_dirs_batch = []

        base_directory = f'./{self.domain}_augmented'
        original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')
        concatenated_dir = os.path.join(base_directory, 'concatenated')
        blended_dir = os.path.join(base_directory, 'blended')

        # Ensure these directories exist
        os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)
        os.makedirs(concatenated_dir, exist_ok=True)
        os.makedirs(blended_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):

            label = self.idx_to_class[label_idx]  # Use folder name as label

            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize(self.resize_shape)
            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['original_resized', method, 'generated', 'fractal', 'concatenated', 'blended']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            # Save original image
            original_img.save(os.path.join(label_dirs['original_resized'], img_filename))

            # compute and save control image
            if method == "canny_image":
                control_image = self.get_canny_image(original_img)
            elif method == "depth_map":
                control_image = self.get_depth_map(original_img)

            control_image.save(os.path.join(label_dirs[method], img_filename))
            # store batch information
            
            original_img_batch.append(original_img)
            control_img_batch.append(control_image)
            img_filename_batch.append(img_filename)
            label_batch.append(label)
            label_dirs_batch.append(label_dirs)

            # batched generate
            is_last_item = (idx == len(self.original_dataset.samples) - 1)
            if len(original_img_batch) == self.batch_size or is_last_item:
                for prompt in self.prompts:

                    # augmented_images =  self.model_handler.generate_images(prompt, img_path, self.num_augmented_images_per_image,
                    #                                           self.guidance_scale)\

                    print(f"prompt: {prompt}, domain: {self.domain}")
                    if prompt == self.domain:
                        continue
                    
                    # prompt engineering
                    prompt_batch = []

                    for k in range(len(original_img_batch)):
                        expanded_prompt = self.expand_prompt_description(prompt)
                        category = label_batch[k]
                        prompt_new = f"{expanded_prompt}, {category}"
                        prompt_batch.append(prompt_new)

                    # utilize model pipeline of ControlNet
                    negative_prompt_batch = [self.negative_prompt for _ in range(len(original_img_batch))]

                    with torch.cuda.amp.autocast():
                        augmented_images = self.model_pipe(prompt_batch,
                                                            negative_prompt = negative_prompt_batch,
                                                            image=original_img_batch,
                                                            control_image=control_img_batch,
                                                            num_inference_steps=25
                                                            #    guess_mode=True, 
                                                            #    guidance_scale=3.0
                                                            ).images

                    # concatenate, blend, and save
                    for i, img in enumerate(augmented_images):
                        img = img.resize(self.resize_shape)
                        generated_img_filename = f"{img_filename_batch[i]}_generated_{prompt}.jpg"
                        img.save(os.path.join(label_dirs_batch[i]['generated'], generated_img_filename))

                        if not self.utils.is_black_image(img):
                            combined_img = self.utils.combine_images(original_img_batch[i], img)
                            concatenated_img_filename = f"{img_filename_batch[i]}_concatenated_{prompt}.jpg"
                            combined_img.save(os.path.join(label_dirs_batch[i]['concatenated'], concatenated_img_filename))

                            random_fractal_img = random.choice(self.fractal_imgs)
                            fractal_img_filename = f"{img_filename_batch[i]}_fractal_{prompt}.jpg"
                            random_fractal_img.save(os.path.join(label_dirs_batch[i]['fractal'], fractal_img_filename))

                            blended_img = self.utils.blend_images_with_resize(combined_img, random_fractal_img)
                            blended_img_filename = f"{img_filename_batch[i]}_blended_{prompt}.jpg"
                            blended_img.save(os.path.join(label_dirs_batch[i]['blended'], blended_img_filename))

                            augmented_data.append((blended_img, label))

                # clear batch data
                original_img_batch.clear()
                control_img_batch.clear()
                img_filename_batch.clear()
                label_batch.clear()
                label_dirs_batch.clear()

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
