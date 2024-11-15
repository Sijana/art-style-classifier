import os
import pandas as pd
import h5py
import tensorflow as tf
from datasets import load_dataset, Dataset
import json
import logging
from tqdm.auto import tqdm
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tqdm.auto import tqdm

class WikiArtDataPreparer:
    def __init__(self, cache_dir='wikiart_dataset_cache'):
        self.cache_dir = cache_dir
        self.image_size = (224, 224)
        self.selected_styles = [
            "Impressionism", "Realism",
            "Cubism", "Art_Nouveau"
        ]
        self.style_id_to_name = {
            7: "Cubism",
            12: "Impressionism",
            21: "Realism",
            3: "Art_Nouveau"
        }
        self.name_to_style_id = {v: k for k, v in self.style_id_to_name.items()}
        self.style_to_index = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(self.selected_styles),
                tf.range(len(self.selected_styles), dtype=tf.int32)
            ),
            -1
        )

    def load_dataset(self, max_images_per_style=1000):
        """
        Load the WikiArt dataset and filter it based on selected styles.
        """
        try:
            logging.info("Loading WikiArt dataset")
            dataset = load_dataset("huggan/wikiart", split="train", cache_dir=self.cache_dir, streaming=True)
            filtered_data = []
            style_counts = {style: 0 for style in self.selected_styles}
            total_images_to_load = max_images_per_style * len(self.selected_styles)
            progress_bar = tqdm(total=total_images_to_load, desc="Loading images")

            for example in dataset:
                style_id = example['style']
                style_name = self.style_id_to_name.get(style_id, None)
                if style_name and style_name in self.selected_styles and style_counts[style_name] < max_images_per_style:
                    example['style'] = style_name  # Update style with name instead of ID
                    filtered_data.append(example)
                    style_counts[style_name] += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({style: count for style, count in style_counts.items()})

                if all(count >= max_images_per_style for count in style_counts.values()):
                    logging.info("Loaded required number of images for all selected styles.")
                    progress_bar.total = sum(style_counts.values())
                    progress_bar.refresh()

                    self.dataset = Dataset.from_dict({
                        'image': [example['image'] for example in filtered_data],
                        'style': [example['style'] for example in filtered_data],
                    })
                    progress_bar.close()
                    logging.info(f"Dataset filtered. Total images: {len(self.dataset)}")
                    logging.info(f"Images per style: {style_counts}")
                    return self.dataset

            # Complete the progress bar if dataset ends before reaching max_images_per_style
            progress_bar.total = sum(style_counts.values())
            progress_bar.refresh()

            self.dataset = Dataset.from_dict({
                'image': [example['image'] for example in filtered_data],
                'style': [example['style'] for example in filtered_data],
            })
            progress_bar.close()
            logging.info(f"Dataset filtered. Total images: {len(self.dataset)}")
            logging.info(f"Images per style: {style_counts}")
            return self.dataset
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise


    @tf.function
    def decode_and_resize_image(self, file_path):
        """
        Decode image from file path and resize it.
        """
        # Read the file
        image_data = tf.io.read_file(file_path)
        # Decode the image data
        image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
        # Ensure the image has static shape
        image.set_shape([None, None, 3])
        # Resize the image
        image = tf.image.resize(image, self.image_size)
        return image

    @tf.function
    def preprocess_image(self, image, label):
        """
        Preprocess the image and label for the TensorFlow dataset.
        """
        try:
            # Normalize pixel values
            image = tf.cast(image, tf.float32) / 255.0

            # One-hot encode the label
            label = tf.one_hot(label, len(self.selected_styles))

            return image, label
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise

    @tf.function
    def map_func(self, file_path, style):
        """
        Mapping function to decode, resize, and preprocess images and labels.
        """
        # Get label index
        label = self.style_to_index.lookup(style)

        # Decode and resize image
        image = self.decode_and_resize_image(file_path)

        # Preprocess image and label
        return self.preprocess_image(image, label)

    def create_tf_dataset(self, batch_size=32, validation_split=0.2):
        """
        Create TensorFlow datasets from the filtered WikiArt dataset with train/validation split

        Args:
            batch_size: Size of batches for training
            validation_split: Fraction of data to use for validation (default: 0.2)

        Returns:
            tuple: (train_dataset, validation_dataset)
        """
        try:
            if not hasattr(self, 'dataset'):
                raise ValueError("Dataset not loaded. Call load_dataset() or load_filtered_dataset() first.")

            # Data augmentation for training
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomBrightness(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ])

            # Calculate split sizes
            dataset_size = len(self.dataset)
            val_size = int(dataset_size * validation_split)
            train_size = dataset_size - val_size

            # Create indices for shuffling
            indices = np.random.permutation(dataset_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Split the dataset
            train_examples = [self.dataset[int(i)] for i in train_indices]
            val_examples = [self.dataset[int(i)] for i in val_indices]

            def generator(examples):
                for example in examples:
                    image_data = example['image']
                    if isinstance(image_data, bytes):
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                    elif isinstance(image_data, str):  # If it is a path, load and convert
                        with open(image_data, 'rb') as img_file:
                            image = Image.open(img_file).convert('RGB')
                    else:
                        # Assume it is a PIL image
                        image = image_data.convert('RGB')

                    image = image.resize((224, 224))
                    image_array = np.array(image)
                    yield image_array, example['style']

            # Create training dataset
            train_dataset = tf.data.Dataset.from_generator(
                lambda: generator(train_examples),
                output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.string)
                )
            )

            # Create validation dataset
            val_dataset = tf.data.Dataset.from_generator(
                lambda: generator(val_examples),
                output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.uint8),
                    tf.TensorSpec(shape=(), dtype=tf.string)
                )
            )

            # Process training dataset
            train_dataset = train_dataset.map(
                lambda image, style: (self.preprocess_image(image, self.style_to_index.lookup(style))),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Apply augmentation only to training dataset
            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            train_dataset = (train_dataset
                            .shuffle(1000)
                            .batch(batch_size)
                            .cache()
                            .prefetch(tf.data.AUTOTUNE))

            # Process validation dataset (no augmentation)
            val_dataset = val_dataset.map(
                lambda image, style: (self.preprocess_image(image, self.style_to_index.lookup(style))),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            val_dataset = (val_dataset
                          .cache()
                          .batch(batch_size)
                          .prefetch(tf.data.AUTOTUNE))

            print("New function")
            return train_dataset, val_dataset

        except Exception as e:
            logging.error(f"Error creating TF dataset: {str(e)}")
            raise


    def save_filtered_dataset(self, filename='filtered_dataset.json'):
        """
        Save the filtered dataset to a JSON file.
        """
        try:
            if not hasattr(self, 'dataset'):
                raise ValueError("Dataset not loaded. Call load_dataset() first.")

            dataset_dict = {
                'image': [],
                'style': self.dataset['style'],
            }

            for img in self.dataset['image']:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                dataset_dict['image'].append(img_str)

            with open(filename, 'w') as f:
                json.dump(dataset_dict, f)

            logging.info(f"Dataset saved to {filename}")

        except Exception as e:
            logging.error(f"Error saving dataset: {str(e)}")
            raise

    def load_filtered_dataset(self, filename='filtered_dataset.json'):
        """
        Load a previously filtered dataset from a JSON file.
        """
        try:
            with open(filename, 'r') as f:
                dataset_dict = json.load(f)

            images = []
            for img_str in dataset_dict['image']:
                img_data = base64.b64decode(img_str)
                img = Image.open(BytesIO(img_data))
                images.append(img)

            self.dataset = Dataset.from_dict({
                'image': images,
                'style': dataset_dict['style'],
            })

            logging.info(f"Dataset loaded from {filename}")
            logging.info(f"Total images: {len(self.dataset)}")

            return self.dataset

        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

class ProgressBarCallback(Callback):
    def __init__(self, epochs, verbose=1):
        super(ProgressBarCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.progress_bar = tqdm(total=self.epochs, desc='Training', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        # Ensure logs is a dictionary and not None
        logs = logs or {}

        # Handle potential None values for metrics
        log_str = ''
        for metric, value in logs.items():
            log_str += f'{metric}: {value:.4f} ' if value is not None else f'{metric}: N/A '

        self.progress_bar.set_postfix_str(log_str)
        self.progress_bar.update(1)

    def on_train_end(self, logs=None):
        self.progress_bar.close()

class ArtStyleClassifier:
    def __init__(self, num_classes, selected_styles, image_size=(224, 224)):
        self.image_size = image_size
        self.num_classes = num_classes
        self.selected_styles = selected_styles
        self.model = self._build_model()
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def _build_model(self):
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(*self.image_size, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=base_model.input, outputs=predictions)

    def train(self, train_dataset, validation_dataset=None, epochs=10, initial_epoch=0):
        progress_callback = ProgressBarCallback(epochs - initial_epoch)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_dataset else 'loss',
            patience=5,
            restore_best_weights=True
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss' if validation_dataset else 'loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=validation_dataset,
            callbacks=[progress_callback, early_stopping, checkpoint],
            verbose=0
        )
        return history

    def predict_single_image(self, image_path):
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.image_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        predictions = self.model.predict(img_array, verbose=0)
        predicted_probabilities = predictions[0]

        # Print probabilities for each style
        for style, probability in zip(self.selected_styles, predicted_probabilities):
            print(f"Style: {style}, Probability: {probability:.4f}")

        # Print overall predicted style
        predicted_style = self.selected_styles[np.argmax(predicted_probabilities)]
        print(f"\nOverall Predicted Style: {predicted_style}")
        return predictions
        

    def predict_batch(self, dataset):
        return self.model.predict(dataset, verbose=0)

    def save_model(self, path):
        if not path.endswith('.keras'):
            path = path + '.keras'
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    @classmethod
    def load_model(cls, path, num_classes, selected_styles):
        if not path.endswith('.keras'):
            path = path + '.keras'
        instance = cls(num_classes, selected_styles)
        instance.model = tf.keras.models.load_model(path)
        return instance

    def print_predictions(self, dataset):
        """Print predictions along with their labels for a dataset."""
        for images, labels in dataset:
            predictions = self.model.predict(images)
            for i in range(len(images)):
                predicted_label = self.selected_styles[np.argmax(predictions[i])]
                true_label = self.selected_styles[np.argmax(labels[i])]
                print(f"Predicted: {predicted_label}, True: {true_label}")