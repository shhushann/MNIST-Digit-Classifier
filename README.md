# MNIST Digit Classifier

A machine learning-based application to classify handwritten digits (0-9) using a Convolutional Neural Network (CNN). The model predicts the digit based on the input and provides a confidence threshold for predictions. If the model is uncertain, it prompts the user to retry or indicates that the input is not a valid digit.

## Description

This project allows users to either upload an image of a handwritten digit or draw it directly on a pixel board. The CNN model, trained on the MNIST dataset, classifies the digit based on the input. If the model is uncertain (based on a set threshold), it requests the user to retry or informs them that the input doesn't match any known patterns.

### Features:
- **Image Upload**: Upload a scanned image of a handwritten digit for classification.
- **Pixel Drawing Board**: Draw a digit directly on a canvas to get real-time predictions.
- **Confidence Threshold**: If the model is unsure about the prediction, it will ask the user to retry or say that it's not a digit.
- **CNN Model**: A deep learning model trained on the MNIST dataset that classifies digits with over 98% accuracy.

### Differentiating Factors:
- **Uncertainty Detection**: The model includes a confidence threshold, ensuring that only reliable predictions are returned.
- **Interactive UI**: Users can either upload images or draw digits on an HTML5 canvas, providing flexibility in input methods.
- **Real-time Feedback**: Immediate feedback and predictions allow users to interact with the model seamlessly.

## Badges

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Installation

To set up the `MNIST-Digit-Classifier` project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/MNIST-Digit-Classifier.git
    cd MNIST-Digit-Classifier
    ```

2. Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    ```

    Activate the virtual environment:
    - **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    - **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run digit_classifier.py
    ```

This will start the web application, and you can access it at `http://localhost:8501` in your browser.

## Usage

### Input
Provide the following data through the web interface:
- **Upload an Image**: Choose a file containing a handwritten digit.
- **Draw on the Canvas**: Use the pixel board to draw a digit (0-9).

### Output
The app will display:
- **Predicted Digit**: The most likely digit identified by the model.
- **Confidence Level**: The probability of the predicted digit.
- **Retry Prompt**: If the model is unsure, it will suggest retrying or indicate that the input doesn't correspond to any known digit.

### Example Input:
- **Uploaded Image**: A photo of the handwritten digit `3`.
- **Drawn Digit**: The user draws a digit `7` on the canvas.

### Example Output:
- **Predicted Digit**: 3 (from uploaded image)
- **Confidence Level**: 97%
- **Predicted Digit**: 7 (from drawing)
- **Confidence Level**: 92%

## Support

If you encounter any issues, feel free to reach out via the following:
- [Create an issue on GitHub](https://github.com/your-username/MNIST-Digit-Classifier/issues)

## Roadmap

- **Future Releases**:
    - Enhance image input handling for better preprocessing.
    - Introduce a larger dataset for better accuracy in recognizing more handwritten digits.
    - Implement model retraining with new data.

## Contributing

Contributions are welcome! If you would like to contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your forked repository (`git push origin feature-branch`).
5. Open a pull request.

Please ensure that your contributions do not break any existing functionality and that all tests pass.

## Authors and Acknowledgments

- **Shushan Gevorgyan** - Creator and Lead Developer
- **TensorFlow/Keras** - Model implementation and training
- **Streamlit** - Interactive app interface
- **HTML5 Canvas** - Interactive drawing interface

## Project Status

This project is actively maintained and open for contributions. Feel free to submit issues or pull requests for improvements.
