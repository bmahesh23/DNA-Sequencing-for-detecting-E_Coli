# DNA-Sequencing-for-detecting-E_Coli
Data Preprocessing and Classification
This notebook demonstrates data preprocessing and classification using a DNA sequence dataset. It includes steps for data loading, cleaning, encoding, and training a neural network classifier.

Overview
Mount Google Drive: Access your Google Drive to save and load files.

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Import Libraries: Essential libraries for data manipulation and machine learning.

python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
Load Data: Retrieve the dataset of DNA sequences from the UCI Machine Learning Repository.

python
Copy code
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
names = ['Class', 'id', 'Sequence']
data = pd.read_csv(url, names=names)
Preprocess Data: Clean and encode the DNA sequences.

Remove unwanted characters.
One-hot encode the sequences.
python
Copy code
classes = data['Class']
sequences = data['Sequence'].str.replace('\t', '')
dic = {i: list(seq) + [cls] for i, (seq, cls) in enumerate(zip(sequences, classes))}
df = pd.DataFrame(dic).transpose()
df.rename(columns={57: 'Class'}, inplace=True)

temp = df.drop(['Class'], axis=1)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(temp)
df1 = enc.transform(temp).toarray()

df_new = pd.DataFrame(df1)
df["Class"] = df["Class"].replace(to_replace=["+"], value=1).replace(to_replace=["-"], value=0)
df_new["Classes"] = df['Class']
Save and Load Encoder: Persist the encoder for future use.

python
Copy code
with open("drive/MyDrive/Dataset/DataScience-Pianalytix-Models/EColi-encoder.pickle", "wb") as f:
    pickle.dump(enc, f)
Alternative Encoding Method: Use one-hot encoding directly with pandas.

python
Copy code
numerical_df = pd.get_dummies(df)
Results and Analysis
Data Visualization: Use matplotlib and seaborn for visualizing the data.
Model Training: Implement and evaluate a neural network classifier using MLPClassifier.
