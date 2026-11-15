# 📱 Mobile Price Prediction Using Artificial Neural Networks (ANN)

## 📌 Project Overview

Mobile Price Prediction is a machine learning web application that predicts the price category of a smartphone based on its hardware specifications and features.

This project uses an **Artificial Neural Network (ANN)** built with TensorFlow and Keras to classify mobile phones into four price categories:

- Class 0 – Low Cost
- Class 1 – Medium Cost
- Class 2 – High Cost
- Class 3 – Very High Cost

The application provides a user-friendly interface where users can enter mobile specifications and receive a predicted price category with a confidence score.

## 🎯 Project Objectives

- Build an Artificial Neural Network for mobile price classification.
- Analyze the relationship between mobile specifications and price categories.
- Evaluate the ANN model using training and validation data.
- Develop a full-stack web application for real-time predictions.
- Deploy the application so users can access it online.

## 📊 Dataset Information

The project uses a mobile price classification dataset.

| Dataset | Records | Columns |
|---|---:|---:|
| Training dataset | 2,000 | 21 |
| Test dataset | 1,000 | 21 |

The training dataset contains 20 input features and one target variable (`price_range`).

The test dataset contains 20 input features and an `id` column. It does not contain actual price labels.

### Input Features

1. Battery Power
2. Bluetooth
3. Clock Speed
4. Dual SIM
5. Front Camera
6. 4G Support
7. Internal Memory
8. Mobile Depth
9. Mobile Weight
10. Number of Cores
11. Primary Camera
12. Pixel Height
13. Pixel Width
14. RAM
15. Screen Height
16. Screen Width
17. Talk Time
18. 3G Support
19. Touch Screen
20. Wi-Fi

### Target Variable

`price_range`

| Class | Price Category |
|---:|---|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

## 🧠 ANN Model Architecture

The project uses a Sequential Artificial Neural Network.

| Layer | Configuration |
|---|---|
| Input Layer | 20 input features |
| Hidden Layer 1 | 64 neurons, ReLU activation |
| Dropout 1 | 0.4 |
| Hidden Layer 2 | 32 neurons, ReLU activation |
| Dropout 2 | 0.3 |
| Output Layer | 4 neurons, Softmax activation |

### Model Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 16 |
| Regularization | L2 (0.01) |
| Early Stopping | Patience = 10 |
| Restore Best Weights | True |

The model uses ReLU activation in the hidden layers and Softmax activation in the output layer to classify smartphones into four price categories.

## 📈 Model Performance

The model was evaluated using a validation dataset containing 400 samples.

### Accuracy and Loss

| Metric | Result |
|---|---:|
| Training Accuracy | 97.875% |
| Validation Accuracy | 96.50% |
| Training Loss | 0.210271 |
| Validation Loss | 0.220017 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Low Cost (0) | 0.99 | 0.97 | 0.98 |
| Medium Cost (1) | 0.94 | 0.96 | 0.95 |
| High Cost (2) | 0.96 | 0.94 | 0.95 |
| Very High Cost (3) | 0.97 | 0.99 | 0.98 |

**Correct predictions:** 386 out of 400 validation samples.

**Validation accuracy: 96.50%.**

*Note: These results are from validation data, not the separate unlabeled test.csv dataset. The accuracy should not be interpreted as a measured result on that test dataset.*

## 🛠️ Technologies Used

### Machine Learning and Deep Learning
- Python
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas

### Backend
- FastAPI
- Uvicorn
- Joblib

### Frontend
- React.js
- JavaScript
- HTML
- CSS
- Vite

### Development and Deployment
- Google Colab – Model training
- Visual Studio Code – Development
- Git and GitHub – Version control
- Render – Backend hosting
- Vercel – Frontend hosting

## 📂 Project Structure

```text
Mobile-Price-Prediction/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── mobile_price_model.keras
│   │   ├── scaler.pkl
│   │   └── feature_columns.json
│   └── venv/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── index.html
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Mobile-Price-Prediction
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 3. Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

Open the application:

```text
http://localhost:5173
```

## 🔄 Application Workflow

1. User enters mobile phone specifications through the React frontend.
2. React sends the input data to the FastAPI backend.
3. FastAPI validates the input values.
4. The saved Scikit-learn scaler transforms the input features.
5. The trained ANN model processes the scaled data.
6. Softmax outputs probabilities for four price categories.
7. The application displays the predicted category and confidence score.

## 💾 Saved Model Files

| File | Purpose |
|---|---|
| `mobile_price_model.keras` | Trained ANN model |
| `scaler.pkl` | Saved feature scaler |
| `feature_columns.json` | Maintains the correct input feature order |

## 🚀 Deployment

The application is designed to be deployed using:

- **GitHub:** Source code repository.
- **Render:** FastAPI backend and ANN model hosting.
- **Vercel:** React frontend hosting.

Deployment links will be added after successful deployment.

## 🔮 Future Improvements

- Add real-world mobile phone price datasets.
- Display approximate prices in Indian Rupees using verified price data.
- Improve the user interface and mobile responsiveness.
- Add visualizations to explain the prediction results.
- Compare ANN performance with other machine learning algorithms.
- Expand the dataset to improve model generalization.

## 👨‍💻 Author

Developed as a Deep Learning project to demonstrate the practical application of Artificial Neural Networks in smartphone price classification.

---

**⭐ If you find this project interesting, consider giving this repository a star!**
