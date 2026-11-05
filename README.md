# Task 1: Data Cleaning & Preprocessing

## Objective
To learn how to clean, preprocess, and prepare raw data for Machine Learning models by handling missing values, encoding categorical variables, scaling features, and removing outliers.

## Tools Used
- **Python**
- **Pandas** & **NumPy** (for data manipulation and mathematical operations)
- **Matplotlib** (for data visualization)
- *Note: Implemented entirely from scratch without Scikit-Learn or Seaborn to demonstrate a strong foundational understanding of the underlying mathematics.*

## Dataset
- **Titanic Dataset** (`titanic.csv`)

---

## Steps Taken

1. **Data Exploration**: Loaded the dataset and inspected data types, basic statistics, and missing value counts using `df.info()` and `df.isnull().sum()`.
2. **Handling Missing Values**: 
   - Imputed missing `Age` values with the **median** (robust to outliers).
   - Imputed missing `Embarked` values with the **mode** (most frequent).
   - Dropped the `Cabin` column due to an excessively high percentage of missing values (>70%).
3. **Categorical Encoding**: 
   - Applied **Label Encoding** for binary variables (e.g., `Sex`: male → 1, female → 0).
   - Applied **One-Hot Encoding** using `pd.get_dummies()` for nominal variables (e.g., `Embarked`), dropping the first column to avoid the dummy variable trap.
4. **Feature Scaling**: Applied **Standardization (Z-score)** manually using the formula $z = \frac{x - \mu}{\sigma}$ to ensure all numerical features contribute equally to future ML models.
5. **Outlier Detection & Removal**: 
   - Visualized the `Fare` column using a **Boxplot**.
   - Calculated the Interquartile Range (IQR) and filtered out rows where `Fare` fell outside the bounds: `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.

---

## Visualizations

### 1. Outlier Detection (Before Removal)
![Boxplot Before](fare_boxplot_before.png)
*Figure 1: Boxplot showing extreme outliers in the 'Fare' column.*

### 2. Outlier Removal (After Filtering)
![Boxplot After](fare_boxplot_after.png)
*Figure 2: Boxplot after applying the IQR method to filter out extreme values.*
