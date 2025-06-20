# === Library Dasar Python ===
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import os

# === Library untuk Pemodelan & Machine Learning ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# === Library untuk Dataset Eksternal ===
from datasets import load_dataset

def otomatisasiPreproc(ds, target):
    
    data = pd.read_csv(ds)
    def preprocessingData(df, kolom, jenis=1, save_dir="Encoder Tersimpan"):
        os.makedirs(save_dir, exist_ok=True)

        if jenis == 1:
            le = LabelEncoder()
            df[kolom] = le.fit_transform(df[kolom])

            # Simpan encoder
            encoder_path = os.path.join(save_dir, f"{kolom}_label_encoder.pkl")
            with open(encoder_path, 'wb') as file:
                pickle.dump(le, file)

            print(f"LabelEncoder untuk '{kolom}' disimpan di: {encoder_path}")

            return df, le
        
        elif jenis == 2:
            df = pd.get_dummies(df, columns=[kolom], prefix=kolom)
            return df, None
        
        else:
            raise ValueError("Parameter 'jenis' harus 1 (Label Encoding) atau 2 (One Hot Encoding)")

    data1, encoder_gender = preprocessingData(data, 'Gender', 1)
    data2, encoder_interest = preprocessingData(data1, 'Interest', 2)
    data3, encoder_personality = preprocessingData(data2, 'Personality', 1)

    y = data2[target]
    X = data2.drop(columns=[target])

    # Membagi data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train.to_csv("dataset/X_train.csv", index=False)
    X_test.to_csv("dataset/X_test.csv", index=False)
    y_train.to_csv("dataset/y_train.csv", index=False)
    y_test.to_csv("dataset/y_test.csv", index=False)

    # Simpan urutan fitur untuk keperluan inference
    feature_order_path = os.path.join("Encoder Tersimpan", "urutanFitur.pkl")
    with open(feature_order_path, 'wb') as f:
        pickle.dump(list(X_train.columns), f)
    print(f"Urutan fitur disimpan di: {feature_order_path}")
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = otomatisasiPreproc("dataset/data.csv","Personality")