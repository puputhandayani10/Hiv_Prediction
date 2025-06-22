import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample

st.title("🧪 Prediksi HIV dengan Random Forest")

import os
if os.path.exists("HIV_dataset.csv"):
    df = pd.read_csv("HIV_dataset.csv")
elif os.path.exists("HIV_dataset.xlsx"):
    df = pd.read_excel("HIV_dataset.xlsx")
else:
    st.error("Dataset tidak ditemukan di repositori. Upload manual atau pastikan file tersedia.")


if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Cuplikan Dataset:")
    st.dataframe(df.head())

    # Sampling
    df_sample = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Label encoding otomatis
    data = df_sample.copy()
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    if 'Result' not in data.columns:
        st.error("Kolom 'Result' tidak ditemukan. Harap pastikan dataset memiliki kolom target bernama 'Result'.")
    else:
        # Pisahkan fitur dan label
        X = data.drop('Result', axis=1)
        y = data['Result']

        # Oversampling
        data_balanced = pd.concat([X, y], axis=1)
        positive = data_balanced[data_balanced['Result'] == 1]
        negative = data_balanced[data_balanced['Result'] == 0]

        positive_upsampled = resample(positive,
                                      replace=True,
                                      n_samples=len(negative),
                                      random_state=42)

        balanced_df = pd.concat([negative, positive_upsampled])
        X_bal = balanced_df.drop('Result', axis=1)
        y_bal = balanced_df['Result']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

        # Train model
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        st.write("### Evaluasi Model:")
        st.write("Akurasi:", round(accuracy_score(y_test, y_pred), 2))
        st.write("Precision:", round(precision_score(y_test, y_pred), 2))
        st.write("Recall:", round(recall_score(y_test, y_pred), 2))
        st.write("F1 Score:", round(f1_score(y_test, y_pred), 2))

        st.write("### Prediksi Data Baru:")
        with st.form("prediction_form"):
            inputs = []
            for col in X.columns:
                val = st.text_input(f"Masukkan nilai untuk '{col}'")
                inputs.append(val)

            submitted = st.form_submit_button("Prediksi")
            if submitted:
                try:
                    input_array = np.array(inputs).astype(float).reshape(1, -1)
                    prediction = clf.predict(input_array)[0]
                    result = "Positif" if prediction == 1 else "Negatif"
                    st.success(f"Hasil Prediksi: {result}")
                except Exception as e:
                    st.error(f"Input tidak valid: {e}")
