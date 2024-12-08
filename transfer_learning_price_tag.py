import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.imagenet_utils import preprocess_input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Paths to the dataset
train_dir = 'price_tags_databse/train'
val_dir = 'price_tags_databse/val'
test_dir = 'price_tags_databse/test'

# Function to load images and labels
def load_images_from_directory(directory, target_size=(224, 224)):
    data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.png', '.jpeg']:
            try:
                img = load_img(filepath, target_size=target_size)
                x = img_to_array(img)
                x = preprocess_input(x)
                data.append({'x': x, 'y': 1})  # Label is 1 for all price tag images
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
    return data

# Load datasets
train_data = load_images_from_directory(train_dir)
val_data = load_images_from_directory(val_dir)
test_data = load_images_from_directory(test_dir)

# Convert data to numpy arrays
def prepare_data(data):
    x = np.array([d['x'] for d in data]).astype('float32') / 255.  # Normalize images
    y = np.array([d['y'] for d in data])  # Labels
    return x, y

x_train, y_train = prepare_data(train_data)
x_val, y_val = prepare_data(val_data)
x_test, y_test = prepare_data(test_data)

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

# Train model
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

# Predict on a new image
def preprocess_single_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

x_with_price = preprocess_single_image('price_tag_test_image.jpeg')
probabilities_with_price = model_new.predict(x_with_price)
print('Price tag image probabilities:', probabilities_with_price)

x_without_price = preprocess_single_image('no_price_tag_test_image.jpeg')
probabilities_without_price = model_new.predict(x_without_price)
print('No price tag image probabilities:', probabilities_without_price)
