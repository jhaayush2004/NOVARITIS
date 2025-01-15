

# Predicting Recruitment Rates in Clinical Trials

### Overview
This project aims to predict the **Study Recruitment Rate (RR)** for clinical trials, a critical metric in the drug development process. By leveraging advanced machine learning models and domain-specific language representations, the solution enhances the prediction process, enabling efficient clinical trial planning and management.

---

### Methodology
1. **Data Preprocessing**:
   - Cleaned textual columns for alphanumeric values.
   - Transformed categorical features using one-hot encoding.
   - Standardized numerical columns with `StandardScaler`.
   - Handled missing values through selective imputation or feature elimination.

2. **Feature Engineering**:
   - Integrated numerical features with textual embeddings extracted via **BioBERT**.
   - Modeled temporal features such as trial duration and completion dates.

3. **Modeling**:
   - Implemented a **Gradient Boosting Regressor (GBM)** for prediction.
   - Fine-tuned hyperparameters using **Bayesian Optimization**.
   - Benchmarked results with alternative models like **LightGBM**.

4. **Evaluation Metrics**:
   - Root Mean Square Error (RMSE): 0.34
   - Mean Absolute Error (MAE): 0.083
   - R² Score: 0.45

---

### Tools & Technologies
- **Transformers (BioBERT)**: For semantic embeddings from biomedical text.
- **PyTorch**: GPU-accelerated computation for embeddings.
- **Scikit-learn**: Model training and evaluation.
- **bayes_opt**: Bayesian hyperparameter optimization.
- **Matplotlib & Seaborn**: Data visualization.
- **Google Colab**: GPU-enabled computation.
- **Pandas & NumPy**: Data handling and numerical computation.

---

### Key Outcomes
- High precision in predicting recruitment rates with low RMSE and MAE.
- Explainability enhanced via SHAP for feature importance analysis.
- Insights into critical factors like trial duration, enrollment numbers, and outcomes.

---

### Challenges & Future Work
- **GPU Limitations**: High-performance GPUs like NVIDIA A100 were utilized, but constraints limited the use of larger models such as GPT-4.
- **Feature Enhancements**: Integration of external datasets for location and sponsor-specific insights.
- **Advanced Temporal Modeling**: Exploring techniques like Temporal Fusion Transformers to improve temporal trend predictions.

---

### Implications
This project provides actionable insights for clinical trial managers to optimize recruitment campaigns and resource allocation, ensuring timely completion and minimizing risks.

**Team Members**: Satyam Kumar, Ayush Shaurya Jha, Raunak Raj, Dhruv Bansal, Kritnandan, Ankita Kumari  

