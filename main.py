import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Import dataset
    df = pd.read_csv("Titanic-Dataset.csv")

    # 2. Explore basic info
    # print("...............head().............")
    # print(df.head())          # First 5 rows
    #
    # print("...............info().............")
    # print(df.info())          # Data types and non-null counts
    #
    # print("...............describe().........")
    # print(df.describe())      # Statistical summary
    #
    # print("...............isnull().sum().....")
    # print(df.isnull().sum())  # Check for missing valuesdef main():

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df = df.drop(columns=['Cabin'])

    # print("...............isnull().sum().....")
    # print(df.isnull().sum())

    # print("...............head().............")
    # print(df.head())          # First 5 rows

    # Label Encoding for binary categories (e.g., 'Sex': male/female -> 1/0)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # One-Hot Encoding for nominal categories with no order (e.g., 'Embarked')
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # print("...............head().............")
    # print(df.head())          # First 5 rows

    # print(".............age, fare head()..........")
    # print(df[['Age', 'Fare']].head())

    # Standardize 'Age' and 'Fare'
    for col in ['Age', 'Fare']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col] - mean_val) / std_val

    # print(".............age, fare head()..........")
    # print(df[['Age', 'Fare']].head())

    # Visualize outliers in 'Fare' using Matplotlib
    plt.figure(figsize=(8, 5))
    plt.boxplot(df['Fare'], vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red'))
    plt.title('Boxplot of Fare (Before Outlier Removal)')
    plt.xlabel('Fare')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

    # Save this plot as a screenshot for your GitHub repo!
    plt.savefig('fare_boxplot_before.png')

    # Calculate IQR manually to remove outliers
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the dataframe
    df_cleaned = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

    print(f"\nOriginal rows: {df.shape[0]}")
    print(f"Rows after removing outliers: {df_cleaned.shape[0]}")

    # 3. Visualize again to confirm outliers are gone
    plt.figure(figsize=(8, 5))
    plt.boxplot(df_cleaned['Fare'], vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='red'))
    plt.title('Boxplot of Fare (After Outlier Removal)')
    plt.xlabel('Fare')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

    # Save the second plot
    plt.savefig('fare_boxplot_after.png')

    df_cleaned.to_csv('titanic_cleaned.csv', index=False)

if __name__ == "__main__":
    main()
