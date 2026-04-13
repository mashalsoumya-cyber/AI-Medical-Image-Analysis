# 🧠 AI-Powered Medical Image Analysis

🚀 A Deep Learning-based web application that analyzes **Chest X-ray images** and predicts whether the patient is **NORMAL** or has **PNEUMONIA**.

---

## 📸 Demo

![Project Demo](project-demo.png)

---

## 🎯 Key Features

✅ Upload Chest X-ray Image  
✅ Predict: **NORMAL / PNEUMONIA**  
✅ Confidence Score Display  
✅ Image Preview in UI  
✅ Clean & Responsive Web Interface  
✅ Model Evaluation with Accuracy  

---

## 🛠️ Tech Stack

| Category       | Technology Used |
|---------------|----------------|
| Language      | Python         |
| ML Framework  | TensorFlow / Keras |
| Backend       | Flask          |
| Image Processing | OpenCV      |
| Data Handling | NumPy, Pandas  |
| Visualization | Matplotlib, Seaborn |
| ML Metrics    | Scikit-learn   |

---

## 📂 Project Structure

AI-Medical-Image-Analysis/
│
├── src/
│ ├── config.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── utils.py
│
├── static/
│ └── style.css
│
├── templates/
│ └── index.html
│
├── app.py
├── main.py
├── requirements.txt
└── README.md


---

## 📊 Model Performance

- ✅ Test Accuracy: **86.38%**
- 📉 Loss & Accuracy Graphs available in `/outputs`

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/mashalsoumya-cyber/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Setup Dataset

⚠️ Dataset is not included in this repo.

Download from Kaggle:
👉 Chest X-ray Pneumonia Dataset

Place it like this:

data/chest_xray/
├── train/
├── val/
└── test/
5️⃣ Train the Model
python main.py
6️⃣ Run Web Application
python app.py

Then open in browser:

http://127.0.0.1:5000
🧪 How It Works
User uploads X-ray image
Image is preprocessed
CNN model predicts class
Result + confidence displayed
💡 Future Improvements

✨ Better UI/UX (premium design)
✨ Drag & Drop Upload
✨ Deploy on Cloud (Render / AWS)
✨ Explainable AI (Heatmaps)
✨ Multi-disease detection

<img width="1908" height="961" alt="Screenshot 2026-04-13 211727" src="https://github.com/user-attachments/assets/55d6fa59-bf00-4194-83ba-6e740a28d7e3" />


