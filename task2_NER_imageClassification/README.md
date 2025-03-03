Named Entity Recognition + Image Classification

📌 Project Overview

This project integrates **Named Entity Recognition (NER)** with **Image Classification** to verify whether an image matches a given textual description.

🔹 Workflow

1️⃣ A user provides a text input (e.g., `"There is a cat in the picture."`)  
2️⃣ A user provides an image (e.g., a picture of a **cat**)  
3️⃣ The **NER model** extracts animal names from the text  
4️⃣ The **Image Classification model** predicts the object in the image  
5️⃣ The system compares both results and returns a **boolean output** (`True` or `False`)

🛠️ Installation & Setup

1️⃣ Clone the Repository

`git clone https://github.com/your_username/internship-test.git`
`cd internship-test/task2_NER_imageClassification`

2️⃣ Set Up a Virtual Environment

# Mac/Linux

`python -m venv venv`
`source venv/bin/activate`

# Windows

`python -m venv venv`
`venv\Scripts\activate`

3️⃣ Install Dependencies

`pip install -r requirements.txt`

🚀 Usage

1️⃣ Run NER Inference on Text

`python scripts/infer_ner.py "There is a dog in the picture."`

Example Output:

Named Entity Recognition Results:
[Dog] -> B-ANIMAL

2️⃣ Run Image Classification Inference

`python scripts/infer_classifier.py --image data/val/dog/1.jpeg`

Example Output:

✅ Prediction: dog

3️⃣ Run the Full Pipeline

`python scripts/pipeline.py`

Example Input:

Enter a text sentence: There is a cat in the picture.
Enter image path: data/val/cat/1039.jpeg

Example Output:

NER Entity Detected: True
Image Classification: cat
✅ Final Boolean Output: True

📌 Model Training

1️⃣ Train Named Entity Recognition Model

`python scripts/train_ner.py`

2️⃣ Train Image Classification Model

`python scripts/train_classifier.py`

🛠️ Troubleshooting

-If models are not found, train them first before running inference.
-Ensure that data/val/ and data/train/ contain correctly structured image datasets.
-If you face library issues, ensure you're using Python 3.8+ and re-install dependencies using:

`pip install --upgrade pip`
`pip install -r requirements.txt`
