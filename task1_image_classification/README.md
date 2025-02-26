# Image Classification Project

## Overview

This project solves the task of **handwritten digit classification** using **three different machine learning models**:

1️⃣ **Random Forest** – A traditional machine learning approach.  
2️⃣ **Feed-Forward Neural Network (FFNN)** – A simple deep learning model.  
3️⃣ **Convolutional Neural Network (CNN)** – A deep learning model optimized for image classification.

Each model is trained on the **MNIST dataset** and compared based on their accuracy.  
Additionally, we **analyze edge cases** to understand where the CNN model struggles.

---

## Solution Explanation

### **How We Solved the Task**

- The project uses the **MNIST dataset**, a collection of 28x28 grayscale images of handwritten digits.
- **Three models** (Random Forest, FFNN, and CNN) were trained and evaluated.
- We implemented **edge case analysis** to analyze model failures.
- The best-performing model, **CNN**, achieved an accuracy of ~99%.

### **Key Steps**

1️⃣ **Data Preprocessing:** Normalized MNIST images for better training.  
2️⃣ **Model Training:** Each model was trained separately.  
3️⃣ **Evaluation:** Models were tested on unseen MNIST images.  
4️⃣ **Edge Case Analysis:** Misclassified images were analyzed.

---

## 🛠️ Setup Instructions

Follow these steps to **install dependencies** and **run the project**.

### **🔹 Prerequisites**

- Python **3.x**
- `pip` package manager

### **🔹 Installation**

1️⃣ **Clone the repository**:

`git clone https://github.com/Hopkomax/internship-test.git`
`cd internship-test/task1_image_classification`

2️⃣ Create and activate a virtual environment:

`python -m venv venv`

# On Windows

`venv\Scripts\activate`

# On macOS/Linux

`source venv/bin/activate`

3️⃣ Install dependencies:

`pip install -r requirements.txt`

How to Run the Project

Running Jupyter Notebook

1️⃣ Open Jupyter Notebook:

`jupyter notebook`

2️⃣ Navigate to the notebooks/demo.ipynb file and open it.

3️⃣ Run the notebook cell by cell to:
-Load the MNIST dataset.
-Train and evaluate Random Forest, FFNN, and CNN.
-Analyze edge cases where the CNN struggles.
