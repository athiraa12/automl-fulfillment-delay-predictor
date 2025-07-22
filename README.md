

# 📦 Fulfillment Delay Predictor with AutoML

A machine learning project that predicts warehouse fulfillment delays using operational data. Built with [TPOT](https://epistasislab.github.io/tpot/), this AutoML-powered pipeline selects the best model for the task and generates actionable recommendations to reduce bottlenecks — inspired by real-world Amazon-style operations.

---

## 🚀 Demo

![image](https://github.com/athiraa12/automl-fulfillment-delay-predictor/blob/main/demo/Screenshot%202025-07-23%20013450.png?raw=true)
---
![image](https://github.com/athiraa12/automl-fulfillment-delay-predictor/blob/main/demo/Screenshot%202025-07-23%20013550.png?raw=true)
---

## 🧠 Project Highlights

- ✅ Built an **AutoML pipeline** using TPOT (LightGBM-based) with ~83% accuracy
- 📊 Simulated warehouse data (orders, workers, pick time, shift hours)
- 🧮 Feature importance and correlation visualizations included
- 🤖 Generates **daily operational recommendations** (e.g., "Add workers", "Extend shifts")
- 💻 Ready for deployment in a Streamlit dashboard

---

## 📁 Dataset

A synthetically generated dataset simulating warehouse operations with the following fields:

| Column Name        | Description                                       |
|--------------------|---------------------------------------------------|
| day                | Day of the week (Mon–Sun)                         |
| orders_received    | Number of orders received that day                |
| workers_available  | Number of workers available during the shift      |
| avg_pick_time      | Avg time to pick one item (in minutes)            |
| shift_hours        | Total shift hours worked                          |
| delay_occurred     | Target variable (1 = Delay, 0 = No delay)         |

> Dataset size: 500 rows × 6 columns

---

## 🧪 How It Works

1. **Data Preprocessing**  
   - Encodes categorical features  
   - Splits into training/testing sets  
   
2. **AutoML Pipeline**  
   - TPOT runs generations of model pipelines  
   - Best model auto-selected (LightGBMClassifier with feature selection)  
   
3. **Prediction + Recommendation Engine**  
   - Predicts if delay will occur  
   - Gives real-time advice to reduce delay

---

## 📊 Visualizations

- Feature importance plot
- Correlation heatmap
- Delay vs. non-delay summaries

---

## 🧠 Sample Recommendation Logic

```python
if workload_ratio > 20 and avg_pick_time > 5:
    return "Add more workers or split workload"
