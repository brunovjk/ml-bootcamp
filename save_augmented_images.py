import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
import numpy as np
from PIL import Image

# Paths
root = 'price_tags'
output_dir = 'augmented_dataset'
os.makedirs(output_dir, exist_ok=True)

# Parameters
target_size = (224, 224)
num_augmented_images_per_original = 5

# Create directories for augmented dataset
os.makedirs(f"{output_dir}/train", exist_ok=True)
os.makedirs(f"{output_dir}/val", exist_ok=True)
os.makedirs(f"{output_dir}/test", exist_ok=True)

# Function to save augmented images
def save_augmented_images(generator, image, label, save_dir, prefix):
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    i = 0
    for batch in generator.flow(img_array, batch_size=1, save_to_dir=save_dir,
                                save_prefix=f"{prefix}_{label}", save_format="jpeg"):
        i += 1
        if i >= num_augmented_images_per_original:
            break

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Load and augment images
images = [os.path.join(root, f) for f in os.listdir(root)
          if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]

if not images:
    raise ValueError("No images found in the dataset. Check your `root` directory.")

# Shuffle images and split into train, val, test
random.shuffle(images)
train_split, val_split = 0.7, 0.15
idx_val = int(train_split * len(images))
idx_test = int((train_split + val_split) * len(images))

train_images = images[:idx_val]
val_images = images[idx_val:idx_test]
test_images = images[idx_test:]

# Save augmented images
for split, image_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
    for img_path in image_list:
        try:
            label = 1  # Since all images are of the same category (price tags)
            img = load_img(img_path, target_size=target_size)
            save_dir = os.path.join(output_dir, split)
            save_augmented_images(datagen, img, label, save_dir, prefix=os.path.splitext(os.path.basename(img_path))[0])
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Confirm completion
print(f"Augmented dataset created in {output_dir}")
