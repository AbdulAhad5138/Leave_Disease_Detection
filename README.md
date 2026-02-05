# 🌿 Plant Disease Detection AI

An advanced AI-powered application for detecting plant diseases from leaf images. Built with **Flask** and **PyTorch**, featuring a modern "Glassmorphism" UI and a tailored **U-Net** segmentation model.

![Plant Disease AI](https://placehold.co/1200x600/10B981/ffffff?text=Plant+Disease+AI+Preview)

## 🚀 Features

*   **Precision AI**: Uses a custom-trained **ResNet34 + U-Net** model (97MB) for pixel-perfect disease segmentation.
*   **Smart Visualization**: Overlays a red mask on the exact diseased areas of the leaf, rather than just a simple "healthy/unhealthy" label.
*   **Modern UI**: Fully responsive, dark-mode interface with glassmorphism design, drag-and-drop uploads, and smooth animations.
*   **Instant Analysis**: Fast inference time powered by PyTorch.

## 🛠️ Tech Stack

*   **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (Vanilla)
*   **Backend**: Flask (Python)
*   **AI/ML**: PyTorch, Segmentation Models PyTorch (SMP), OpenCV
*   **Dataset**: Custom dataset from Roboflow

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/AbdulAhad5138/Leave_Disease_Detection.git
    cd Leave_Disease_Detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Model Setup**
    *   Ensure the trained model file `best_leaf_model.pth` is in the root directory.
    *   *Note: If cloning, the model file is included but large (93MB).*

4.  **Run the App**
    ```bash
    python app.py
    ```

5.  **Access the Interface**
    *   Open your browser and visit: `http://127.0.0.1:5000`

## 🖥️ Usage

1.  Click **"Browse Files"** or drag and drop a leaf image into the upload zone.
2.  Wait for the **AI Diagnostics** to process the image.
3.  View the side-by-side comparison:
    *   **Left**: Original Image.
    *   **Right**: AI Prediction (Red areas indicate disease).

## 📂 Project Structure

```
Leave_Disease_Detection/
├── app.py                 # Main Flask Application
├── best_leaf_model.pth    # Trained PyTorch Model
├── requirements.txt       # Python Dependencies
├── static/
│   ├── css/               # Stylesheets
│   ├── js/                # Frontend Logic
│   └── uploads/           # Temp storage for uploads
└── templates/
    └── index.html         # Main HTML Template
```

## 🤝 Acknowledgements

*   **Roboflow**: For providing the tools to manage the dataset.
*   **Segmentation Models PyTorch**: For the excellent U-Net implementation.

---
*Developed by [Abdul Ahad](https://github.com/AbdulAhad5138)*
