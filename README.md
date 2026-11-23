# ✳️ DataMind AI

**DataMind AI** is an intelligent, end-to-end Data Exploration and Machine Learning web application. It is designed to make data analysis and predictive modeling accessible to everyone, without the need to write a single line of code.

Built with **Python**, **Streamlit**, and **Scikit-learn**, this app allows users to upload any CSV dataset, explore its statistical properties, visualize correlations, and train a Machine Learning model instantly.

---

## 🚀 Key Features

- **📊 Comprehensive Data Exploration:** Upload any CSV file and instantly view records, missing values, duplicates, and column data types.
- **📈 Interactive Visualizations:** Automatically generates distributions (histograms) for numeric columns and bar charts for categorical columns.
- **🔥 Correlation Heatmaps:** Visualizes how different features in the dataset are related to one another.
- **🧠 Smart ML Studio:** 
  - Automatically detects whether the target variable requires **Regression** (Numbers) or **Classification** (Categories).
  - Applies a **Random Forest** algorithm with a robust pipeline (handling missing values via `SimpleImputer` and encoding categorical variables via `OneHotEncoder`).
- **🎯 Model Evaluation:** Displays R² Score, MAE, RMSE for regression, and Accuracy for classification.
- **📥 Export Results:** Allows users to download their model predictions as a CSV file.
- **🎨 Premium UI/UX:** Features a fully responsive, custom-designed interface with smooth animations and a modern color palette.

---

## 🛠️ Tech Stack

- **Frontend / Web Framework:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest, Pipelines, ColumnTransformers)
- **Data Visualization:** Altair
- **Styling:** Custom CSS (integrated within Streamlit)

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dikshitaaa/DataMindAI.git
   cd DataMindAI

   # ✳️ DataMind AI

**🚀 Live Demo:** [https://datamind-lab.streamlit.app/](https://datamind-lab.streamlit.app/)

**DataMind AI** is an intelligent, end-to-end Data Exploration and Machine Learning web application...
