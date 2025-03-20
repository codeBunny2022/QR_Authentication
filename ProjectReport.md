**QR Code Authentication Project Report**


---

# **1. Introduction**

In this project, we implemented a **QR Code Authentication System** to differentiate between **genuine and counterfeit QR codes**. The system uses a combination of **Machine Learning (Random Forest, SVM) and Deep Learning (CNN)** to classify QR codes as either **First Print (Real)** or **Second Print (Fake)**.

The main objectives of this project are:

* To develop a robust authentication system for QR codes.
* To compare the performance of traditional machine learning models against a deep learning model.
* To deploy the solution via a **Flask-based API** and an **interactive React UI**.


---

# **2. Methodology**

## **2.1 Data Collection & Preprocessing**

* **Dataset**: The dataset consists of QR code images categorized as **First Print (Real)** and **Second Print (Fake)**.
* **Preprocessing Steps**:
  * Converted images to grayscale.
  * Resized all images to **256x256 pixels**.
  * Applied **Histogram Equalization** and **Gaussian Blur** for noise reduction.

## **2.2 Feature Extraction**

We used **two feature extraction techniques** for traditional ML models:

* **HOG (Histogram of Oriented Gradients)**: Captures edge and texture information.
* **LBP (Local Binary Patterns)**: Extracts texture-based features.

## **2.3 Model Selection & Training**

### **Traditional Machine Learning Models**

* **Random Forest**: A tree-based ensemble model.
* **Support Vector Machine (SVM)**: A linear classifier with a kernel trick.

### **Deep Learning Model**

* **CNN (Convolutional Neural Network)**: A sequential model with **Batch Normalization, MaxPooling, and Dropout layers** for better generalization.
* **Data Augmentation** was applied to improve model robustness.

## **2.4 Deployment**

* **Flask API** to handle predictions.
* **React Frontend** for user interaction, file upload, and displaying results.


---

# **3. Experiments & Results**

## **3.1 Model Performance Metrics**

The models were evaluated using **Accuracy, Precision, Recall, and F1-score**.

| Model | Accuracy | Precision | Recall | F1-score |
|----|----|----|----|----|
| Random Forest | **99%** | 0.99 | 0.99 | 0.99 |
| SVM | **100%** | 1.00 | 1.00 | 1.00 |
| CNN | **61%** | 0.78 | 0.61 | 0.54 |

## **3.2 Confusion Matrix**

Below is the confusion matrix for each model:

### **Random Forest**

| Actual / Predicted | First Print (0) | Second Print (1) |
|----|----|----|
| First Print (0) | 99 | 1 |
| Second Print (1) | 1 | 99 |

### **SVM**

| Actual / Predicted | First Print (0) | Second Print (1) |
|----|----|----|
| First Print (0) | 100 | 0 |
| Second Print (1) | 0 | 100 |

### **CNN**

| Actual / Predicted | First Print (0) | Second Print (1) |
|----|----|----|
| First Print (0) | 56 | 44 |
| Second Print (1) | 78 | 22 |

## **3.3 Training Metrics for CNN**

* **Epochs**: 10
* **Final Training Accuracy**: 94.2%
* **Final Validation Accuracy**: 87.5%
* **Loss Curve**: The model suffered from **overfitting**, leading to suboptimal performance on test data.


---

# **4. Conclusion & Future Work**

## **4.1 Key Findings**

* **SVM achieved 100% accuracy**, making it the best model for this problem.
* **Random Forest performed very well**, with 99% accuracy.
* **CNN underperformed** due to limited training data and potential overfitting.

## **4.2 Future Improvements**

* **Improve CNN Performance** using **Transfer Learning (ResNet50, EfficientNet)**.
* **Expand the Dataset** to enhance model generalization.
* **Deploy the Model Online** using AWS/GCP for live authentication.


---

I did successfully developed a **QR Code Authentication System** with a strong **SVM-based model** and an **interactive UI** in **1 day**. 🚀