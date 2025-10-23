<<<<<<< HEAD
# 🚀 Spectra_AI_Anomaly_Detection  
**Prompt-Level Anomaly Detection for AI Safety using Mahalanobis Distance, Ledoit–Wolf Covariance, and Bayesian Inference**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Scikit--learn%20%7C%20NumPy%20%7C%20SciPy-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## 📘 Overview
This project was developed as part of the **Spectra AI Mini Challenge**, focusing on building an **interpretable anomaly detection system** for identifying unsafe or malicious AI prompts.  
It leverages:
- **Mahalanobis Distance** for measuring prompt embedding deviations,  
- **Ledoit–Wolf Covariance Estimation** for robust covariance regularization, and  
- **Bayesian Posterior Probability** for reliability estimation.

---

## 🧠 Methodology

### 1️⃣ Feature Extraction  
Prompt embeddings are obtained from a pre-trained LLM encoder and stored as `.npy` files.  
- Normal and anomalous prompts are represented in 768-dimensional feature space.

### 2️⃣ Covariance Regularization  
The **Ledoit–Wolf estimator** is applied to compute a well-conditioned covariance matrix ensuring numerical stability.

### 3️⃣ Mahalanobis Distance Computation  
Each embedding’s deviation from the normal distribution is measured using the Mahalanobis distance.

### 4️⃣ Chi-Square Probability Mapping  
Distances are mapped to p-values using a chi-square distribution.  
Prompts with **p < 0.01** are flagged as anomalies.

### 5️⃣ Bayesian Posterior Reliability  
Posterior probability `P(malicious | flagged)` is computed to evaluate real-world reliability under various prior assumptions.

---

## 📊 Results Summary

| **Metric** | **Value** | **Interpretation** |
|-------------|------------|--------------------|
| Total Prompts | 1080 | Complete dataset size |
| Normal Prompts | 1000 | Safe inputs |
| Anomalous Prompts | 80 | Unsafe/injected inputs |
| Flagged Prompts (p < 0.01) | 88 | 80 anomalies + 8 false positives |
| ROC AUC | **1.000** | Perfect class separation |
| Accuracy | **99.3%** | Excellent model reliability |
| Precision (Anomalies) | 90.9% | Minimal false positives |
| Recall (Anomalies) | 100% | All anomalies detected |
| F1-score | 0.95 | Balanced performance |

---

## 📈 Visualizations

### 🧩 Distribution of Embedding Feature Values
![Embedding Distribution](images/Distribution%20of%20Embedding%20Feature%20Values.png)

---

### 📉 Histogram of Mahalanobis Distances
![Mahalanobis Histogram](images/Histogram%20of%20Mahalanobis%20Distances.png)

---

### 🎯 ROC Curve (Mahalanobis)
![ROC Curve](images/ROC%20Curve%20(Mahalanobis).png)

---

## 🧮 Key Findings
- The system achieved **ROC AUC = 1.0** and **Accuracy = 99.3%**.  
- **False Positive Rate:** Only 0.8% of safe prompts were wrongly flagged.  
- **True Positive Rate:** 100% of anomalies were correctly detected.  
- **Bayesian reliability** exceeded 93% for realistic priors, showing robustness under real-world conditions.  
- Framework is interpretable, mathematically sound, and deployment-ready.

---

## 📂 Project Structure

```
Spectra_AI_Anomaly_Detection/
│
├── data/
│   ├── embeddings.npy
│   ├── normal_embeddings.npy
│   ├── anomalous_embeddings.npy
│   └── labels.npy
│
├── images/
│   ├── Distribution of Embedding Feature Values.png
│   ├── Histogram of Mahalanobis Distances.png
│   └── ROC Curve (Mahalanobis).png
│
├── docs/
│   └── reference_notes.txt
│
├── Spectra_AI_Mini_Challenge_Atharv_Kulkarni.ipynb
└── README.md
```

---

## 🧰 Technologies Used
- **Language:** Python 3.10  
- **Libraries:** NumPy, SciPy, Scikit-learn, Matplotlib  
- **Concepts:** Covariance Regularization, Mahalanobis Distance, Bayesian Probability, Statistical Inference  

---

## 🧩 How to Run
```bash
# Clone this repository
git clone https://github.com/Kulkarni-ui/Spectra_AI_Anomaly_Detection.git

# Navigate to the directory
cd Spectra_AI_Anomaly_Detection

# Open Jupyter Notebook
jupyter notebook Spectra_AI_Mini_Challenge_Atharv_Kulkarni.ipynb
```

---

## 🧠 Author
**Atharv Kulkarni**  
B.Tech Artificial Intelligence & Machine Learning  
Symbiosis Institute of Technology, Pune  
📧 [Contact via GitHub](https://github.com/Kulkarni-ui)

---

## 🪶 References
1. Ledoit, O., & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix.*  
2. Mahalanobis, P. C. (1936). *On the Generalized Distance in Statistics.*  
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*  
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective.*

---

⭐ *If you found this project insightful, don’t forget to star the repository!*
=======
# Spectra_AI_Anomaly_Detection
Spectra AI Mini Challenge — Prompt-level anomaly detection using Mahalanobis distance, Ledoit–Wolf covariance, and Bayesian inference for AI safety
>>>>>>> 9e502e967101c3188757088244f6cfcdc3dfb48d
