import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import os 
import requests
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

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

def load_model():
    # Adjust these based on your trained model
    # URL of the model file from the GitHub release 
    model_url = "https://github.com/Sijana/art-style-classifier/releases/download/Model-keras/cnn_900_model.h5.keras" 
    # # Download the model file if it doesn't already exist 
    model_path = 'cnn_900_model.h5.keras' 
    if not os.path.exists(model_path): 
        with st.spinner("Downloading model..."): 
            response = requests.get(model_url) 
            with open(model_path, 'wb') as file: 
                file.write(response.content)
    
    STYLE_NAMES = ["Impressionism", "Realism", "Cubism", "Art_Nouveau"] # how can i make this dynamic?
    return ArtStyleClassifier.load_model(model_path, len(STYLE_NAMES), STYLE_NAMES), STYLE_NAMES

def main():
    # Load model
    model, style_names = load_model()
    formatted_styles = ', '.join(style_names)
    
    st.title('Art Style Classifier')
    st.write("Art Style Classifier using Convoluted Neural Network using ResNet50 and WikiArts database.")
    st.write(f'Upload an image to classify its artistic style. Current styles: {formatted_styles}.')
    

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    
    st.write("© Sijana Mamos 2024")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Make prediction
        predictions = model.predict_single_image(uploaded_file)
        print(predictions)
        
        # Display results
        st.write("## Predicted Style Probabilities")
        
        # Create a bar chart of probabilities
        probabilities = predictions[0]
        chart_data = pd.DataFrame({
            'Style': style_names,
            'Probability': probabilities
        })
        
        st.bar_chart(chart_data.set_index('Style'))
        
        # Display top prediction
        top_prediction = style_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        st.write(f"### Top Prediction: {top_prediction}")
        st.write(f"Confidence: {confidence:.2%}")
        

if __name__ == '__main__':
    main()
