# 📞 Telecom Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to churn (cancel their service), enabling proactive retention strategies and reducing customer acquisition costs.

## 🎯 Problem Statement

Customer churn costs telecom companies billions annually. Acquiring a new customer costs 5-7x more than retaining an existing one. This project builds a predictive model to identify at-risk customers before they leave, allowing companies to take preventive action.

## 🚀 Key Features

📈 **Predictive Analytics**: Machine learning classification to identify churn patterns  
🤖 **Multiple Algorithms**: Comparison of Logistic Regression, Decision Tree, Random Forest, and XGBoost  
⚖️ **Handles Imbalanced Data**: SMOTE technique for better minority class prediction  
📊 **Feature Importance Analysis**: Identifies key drivers of customer churn  
🎯 **Business Insights**: Actionable recommendations for customer retention  
🌐 **Flask API**: Production-ready REST API for real-time predictions

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 78.8% |
| **Precision** | 82% |
| **Recall** | 79% |
| **F1-Score** | 80% |
| **ROC-AUC** | 85% |

## 🛠 Tech Stack

**Language:** Python 3.8+

**Libraries:**
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Flask, Pickle

**Tools:** Jupyter Notebook, Git, VS Code

## 💾 Dataset

**Source:** Telco Customer Churn Dataset  
**Samples:** 7,043 customers  
**Features:** 20 (demographics, services, account info)  
**Target:** Churn (Yes/No)

### Key Features:
- Customer demographics (gender, age, dependents)
- Service subscriptions (phone, internet, streaming)
- Account information (tenure, contract type, payment method)
- Billing details (monthly charges, total charges)

## 🔍 Key Insights

### Top Churn Predictors:
1. **Contract Type**: Month-to-month contracts have 3x higher churn rate
2. **Tenure**: Customers with <6 months are high-risk
3. **Payment Method**: Electronic check users churn more frequently
4. **Monthly Charges**: Higher charges correlate with increased churn
5. **Tech Support**: No tech support subscription increases churn probability

### Business Recommendations:
✅ Target month-to-month customers with long-term contract incentives  
✅ Implement robust onboarding program for first 6 months  
✅ Promote automatic payment methods with discounts  
✅ Offer bundled services to justify higher pricing  
✅ Provide proactive tech support to at-risk segments

## 💻 Installation & Usage

### Prerequisites
```bash
python >= 3.8
pip
```

### Setup
```bash
# Clone the repository
git clone https://github.com/vanshthiske/Projects.git
cd "Projects/4 Telecom Customer Churn Prediction"

# Install dependencies
pip install -r requirements.txt
```

### Run Jupyter Notebook
```bash
jupyter notebook "Telecom Customer Churn Prediction .ipynb"
```

### Run Flask API
```bash
python app.py
```

API will be available at `http://localhost:5000`

### Make Predictions
```python
import requests

# Sample customer data
data = {
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    # ... other features
}

# POST request
response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
```

## 📁 Project Structure

```
4 Telecom Customer Churn Prediction/
│
├── Telco Customer Churn.csv          # Dataset
├── Telecom Customer Churn Prediction .ipynb  # Main analysis notebook
├── app.py                             # Flask API
├── model.pkl                          # Trained model
├── encoders.pkl                       # Label encoders
├── requirements.txt                   # Dependencies
└── README.md                          # Documentation
```

## 🧠 Methodology

### 1. **Data Preprocessing**
- Handled missing values in TotalCharges
- Encoded categorical variables (Label Encoding, One-Hot Encoding)
- Feature scaling for numerical features
- Train-test split (80-20)

### 2. **Exploratory Data Analysis**
- Churn rate analysis: 26.5% overall churn
- Feature correlation analysis
- Distribution analysis across customer segments
- Identified class imbalance (73.5% vs 26.5%)

### 3. **Model Development**
Trained and compared multiple algorithms:
- Logistic Regression (Baseline)
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### 4. **Handling Imbalance**
- Applied SMOTE (Synthetic Minority Over-sampling Technique)
- Improved recall for churn class from 65% to 79%

### 5. **Model Evaluation**
- Cross-validation (5-fold)
- Confusion matrix analysis
- Feature importance ranking
- ROC-AUC curve

### 6. **Deployment**
- Serialized best model with Pickle
- Built Flask REST API
- Created prediction endpoint

## 📊 Visualizations

The project includes:
- Confusion Matrix
- Feature Importance Chart
- Model Comparison Bar Charts
- Churn Distribution Analysis
- ROC Curve
- Correlation Heatmap

*(Add your generated images here after running the notebook)*

## 🚀 Future Improvements

☐ Implement Deep Learning models (Neural Networks)  
☐ Add customer lifetime value (CLV) prediction  
☐ Build interactive dashboard with Streamlit  
☐ Deploy on cloud (AWS/Azure/Heroku)  
☐ Add A/B testing framework for retention strategies  
☐ Integrate real-time data pipeline  
☐ Implement explainable AI (SHAP/LIME) for model interpretability

## 📈 Business Impact

**Potential ROI:**
- **Current Churn Rate:** 26.5%
- **Projected Reduction:** 5-7% (20-25% improvement)
- **Cost Savings:** ₹[X] per retained customer
- **Annual Revenue Protection:** ₹[Y] million

## 📝 Requirements

```txt
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
imbalanced-learn==0.11.0
matplotlib==3.7.1
seaborn==0.12.2
flask==2.3.2
pickle-mixin==1.0.2
```

## 👨‍💻 Author

**Vansh Vijay Thiske**  
🔗 [GitHub](https://github.com/vanshthiske) | 💼 [LinkedIn](https://linkedin.com/in/vansh-thiske)

Feel free to reach out for collaborations or questions!

## 📝 License

This project is open source and available for educational purposes.

---

⭐ **If you found this project helpful, please give it a star!**  
🐛 **Found a bug? Open an issue!**  
🚀 **Want to contribute? Pull requests are welcome!**
