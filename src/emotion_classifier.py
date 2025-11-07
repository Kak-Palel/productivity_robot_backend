from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import cv2

class EmotionClassifier:
    def __init__(self, model_name='abhilash88/face-emotion-detection'):
        # Load model and processor
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def classify(self, image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"Loaded image (cv2 numpy): {image_path} shape={image.shape} dtype={image.dtype}")
        inputs = self.processor(image, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()

        # Emotion classes
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotions[predicted_class]
        confidence = predictions[0][predicted_class].item()

        return predicted_emotion, confidence, predicted_class

if __name__ == "__main__":
    classifier = EmotionClassifier()
    image_path = 'man.png'  # Replace with your image path
    emotion, confidence, class_id = classifier.classify(image_path)
    print(f"Predicted Emotion: {emotion} ({confidence:.2f}), Class ID: {class_id}")