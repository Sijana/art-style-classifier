import streamlit as st
import numpy as np
from PIL import Image
from modules import helpers
import pandas as pd

def load_model():
    # Adjust these based on your trained model
    STYLE_NAMES = ["Impressionism", "Realism", "Cubism", "Art_Nouveau"] # how can i make this dynamic?
    return helpers.ArtStyleClassifier.load_model('cnn_900_model.h5.keras', len(STYLE_NAMES), STYLE_NAMES), STYLE_NAMES

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