import os
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array  # type: ignore
from keras.applications.imagenet_utils import preprocess_input  # type: ignore
from keras.models import Model  # type: ignore
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D  # type: ignore
from keras.applications import VGG16  # type: ignore
from keras.callbacks import ReduceLROnPlateau, EarlyStopping  # type: ignore
from keras.regularizers import l2  # type: ignore

# Function to load and preprocess images
def get_image(path):
    img = load_img(path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Dataset path
root = 'price_tags'

# Load all images as a single category
images = [os.path.join(root, f) for f in os.listdir(root)
          if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]

# Ensure images were found
if not images:
    raise ValueError("No images found in the dataset. Check your `root` directory.")

# Create data and labels (all labels = 1 since it's a single category)
data = []
for img_path in images:
    try:
        img, x = get_image(img_path)
        data.append({'x': np.array(x[0]), 'y': 1})  # Label as 1 (price tag present)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

# Shuffle and split data
random.shuffle(data)
train_split, val_split = 0.7, 0.15
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))

train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

# Separate features and labels
x_train = np.array([t['x'] for t in train]).astype('float32') / 255.
y_train = np.array([t['y'] for t in train])

x_val = np.array([t['x'] for t in val]).astype('float32') / 255.
y_val = np.array([t['y'] for t in val])

x_test = np.array([t['x'] for t in test]).astype('float32') / 255.
y_test = np.array([t['y'] for t in test])

# Check dataset shapes
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Build model using VGG16
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(vgg.output)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)
model_new = Model(inputs=vgg.input, outputs=out)

# Unfreeze the last few layers of VGG16 progressively
for layer in vgg.layers[:-6]:
    layer.trainable = False

# Compile model
model_new.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Callbacks for better training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
]

# Train model using data generators
model_new.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# Evaluate model
loss, accuracy = model_new.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Predict on a new image with a price tag
def preprocess_single_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

x_with_price = preprocess_single_image('price_tag_test_image.jpeg')
probabilities_with_price = model_new.predict(x_with_price)
print('Price tag image probabilities:', probabilities_with_price)

# Predict on a new image without a price tag
x_without_price = preprocess_single_image('no_price_tag_test_image.jpeg')
probabilities_without_price = model_new.predict(x_without_price)
print('No price tag image probabilities:', probabilities_without_price)
