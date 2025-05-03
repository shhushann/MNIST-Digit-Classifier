import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas

# ---------------------------
# Define the CNN model
# ---------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust based on your trained model
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten to match fc1's input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model = DigitCNN()
model.load_state_dict(torch.load('digit_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# ---------------------------
# Image preprocessing
# ---------------------------
def center_image(img):
    img_array = np.array(img.convert('L'))
    threshold = 20
    bin_img = img_array < (255 - threshold)

    if bin_img.sum() == 0:
        return Image.fromarray(img_array)

    coords = np.column_stack(np.where(bin_img))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img_array[y0:y1+1, x0:x1+1]
    side = max(cropped.shape)
    new_img = Image.new('L', (side, side), color=255)
    paste_x = (side - cropped.shape[1]) // 2
    paste_y = (side - cropped.shape[0]) // 2

    cropped_img = Image.fromarray(cropped)
    new_img.paste(cropped_img, (paste_x, paste_y))
    return new_img

def preprocess_image(image):
    image = center_image(image)
    image = image.resize((28, 28), Image.LANCZOS)  # Resize to 28x28 for the model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing to the same values as in training
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ---------------------------
# Prediction function
# ---------------------------
def predict_digit(image):
    try:
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            confidence = confidence.item()
            prediction = predicted.item()

        if confidence >= 0.75:
            return f"✅ Predicted Digit: {prediction} (Confidence: {confidence:.2%})"
        elif confidence >= 0.50:
            return f"⚠️ Not confident. Please try again. (Top guess: {prediction}, Confidence: {confidence:.2%})"
        else:
            return f"❌ Doesn't seem like a digit. (Confidence: {confidence:.2%})"

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Digit Classifier", page_icon="🧠")
st.title("🧠 Digit Classifier")
st.write("Upload or draw a digit (0–9) to see the model's prediction. The image will be centered before classification.")

# Option 1: File uploader
uploaded_file = st.file_uploader("📂 Upload a digit image", type=["jpg", "jpeg", "png"])

# Option 2: Drawing Canvas
st.markdown("🎨 Or draw a digit below:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

image_to_predict = None

if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file).convert('RGB')
    st.image(image_to_predict, caption="Uploaded Image", use_column_width=True)

elif canvas_result.image_data is not None:
    # Convert canvas NumPy image to PIL
    img_array = (255 - canvas_result.image_data[:, :, 0]).astype(np.uint8)  # convert black on white
    image_to_predict = Image.fromarray(img_array)

if image_to_predict:
    centered_img = center_image(image_to_predict)
    st.image(centered_img, caption="Centered Image (used for prediction)", use_column_width=True)

    if st.button("Predict"):
        result = predict_digit(image_to_predict)
        st.subheader("Result")
        st.success(result)
