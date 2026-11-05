import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Import dataset
    df = pd.read_csv("Titanic-Dataset.csv")

    # 2. Explore basic info
    print("...............head().............")
    print(df.head())          # First 5 rows

    print("...............info().............")
    print(df.info())          # Data types and non-null counts

    print("...............describe().........")
    print(df.describe())      # Statistical summary

    print("...............isnull().sum().....")
    print(df.isnull().sum())  # Check for missing valuesdef main():

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df = df.drop(columns=['Cabin'])

    print("...............isnull().sum().....")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
