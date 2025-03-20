# **QR Code Authentication System**

This project implements a **QR Code Authentication System** that classifies QR codes as **First Print (Real)** or **Second Print (Fake)** using **Machine Learning (Random Forest, SVM) and Deep Learning (CNN)**.


## **1. Project Structure**

```
QR_Authentication/
│── backend/                # Flask API for predictions
│   ├── app.py              # Main backend script
│   ├── uploads/            # Stores uploaded images (temporary)
│── frontend/               # React-based user interface
│   ├── src/                # Frontend source files
│── models/                 # Trained ML models
│── data/                   # Dataset (First Print & Second Print QR Codes)
│── src/                    # Core ML & DL scripts
│   ├── config.py           # Configuration file
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── deep_learning.py
│   ├── inference.py
│   ├── traditional_model.py
│── requirements.txt        # Dependencies
│── README.md               # Documentation
```


## **2. Installation & Setup**

### **2.1 Prerequisites**

Ensure you have the following installed:

* **Python 3.8+**
* **Node.js & npm** (for frontend)
* **Virtual Environment (Recommended)**

### **2.2 Install Dependencies**

```sh
pip install -r requirements.txt
```


## **3. Model Training**

## Please make sure that the ( models/ ) directory exist.

**3.1 Train Traditional ML Models**

```sh
python3 src/traditional_model.py
```

This will train **Random Forest & SVM** and save models in the `models/` folder.

### **3.2 Train Deep Learning Model (CNN)**

```sh
python3 src/deep_learning.py
```

This will train a **CNN model** and save it in `models/cnn_model.h5`.


## **4. Running the Backend (Flask API)**

### **4.1 Start the Flask Server**

```sh
cd backend
python3 app.py
```

✅ Flask API should now be running on **<http://127.0.0.1:5000>**.

### **4.2 Test the API with cURL**

```sh
curl -X POST -F "file=@test_image.png" http://127.0.0.1:5000/predict
```


## **5. Running the Frontend (React UI)**

### **5.1 Navigate to Frontend Directory**

```sh
cd frontend
```

### **5.2 Install Dependencies**

```sh
npm install
```

### **5.3 Start the React Application**

```sh
npm start
```

✅ React UI should now be available at **<http://localhost:3000>**.


## **6. Testing the Application**

### **6.1 Upload an Image via UI**



1. Open **<http://localhost:3000>** in your browser.
2. Click **Upload & Predict** after selecting a QR code image.
3. View the **model predictions & final decision**.

### **6.2 Check Logs in Backend**

Run the following to view logs:

```sh
tail -f backend/logs/app.log
```



## **7. Future Improvements**

* We can implement **Transfer Learning (ResNet50, EfficientNet)** to improve CNN accuracy.
* Expand the dataset to enhance model generalization.
* Deploy the model online using AWS/GCP for live authentication.


