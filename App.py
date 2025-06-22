import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Judul Aplikasi
st.title("🧪 Prediksi HIV dengan Random Forest")
st.write("📥 Silakan unggah file dataset HIV dalam format .csv atau .xlsx")

# Upload file
uploaded_file = st.file_uploader("Unggah Dataset HIV", type=["csv", "xlsx"])

try:
    if uploaded_file is not None:
        # Baca file berdasarkan ekstensi
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        # Tampilkan cuplikan dataset
        st.subheader("Cuplikan Dataset")
        st.dataframe(df.head())

        # Ambil sampel 20% dari data
        df_sample = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

        # Deteksi otomatis nama kolom target
        possible_targets = ['Result', 'hiv_result']
        target_column = next((col for col in df_sample.columns if col in possible_targets), None)

        if not target_column:
            st.error("Kolom target 'Result' atau 'hiv_result' tidak ditemukan dalam dataset.")
        else:
            # Label encoding untuk kolom kategorikal (kecuali target)
            data = df_sample.copy()
            label_encoders = {}
            for col in data.columns:
                if data[col].dtype == 'object' and col != target_column:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    label_encoders[col] = le

            # Konversi nilai target POSITIVE/NEGATIVE ke 1/0
            data[target_column] = data[target_column].map({'NEGATIVE': 0, 'POSITIVE': 1})

            # Pisahkan fitur dan label
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            # Oversampling kelas minoritas (jika perlu)
            data_balanced = pd.concat([X, y], axis=1)
            positive = data_balanced[data_balanced[target_column] == 1]
            negative = data_balanced[data_balanced[target_column] == 0]

            positive_upsampled = resample(
                positive,
                replace=True,
                n_samples=len(negative),
                random_state=42
            )

            balanced_df = pd.concat([negative, positive_upsampled])
            X_bal = balanced_df.drop(target_column, axis=1)
            y_bal = balanced_df[target_column]

            # Split data latih dan uji
            X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

            # Latih model Random Forest
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Prediksi dan evaluasi
            y_pred = clf.predict(X_test)

            st.subheader("Evaluasi Model")
            st.metric("Akurasi", round(accuracy_score(y_test, y_pred), 2))
            st.metric("Precision", round(precision_score(y_test, y_pred), 2))
            st.metric("Recall", round(recall_score(y_test, y_pred), 2))
            st.metric("F1 Score", round(f1_score(y_test, y_pred), 2))

            # Form prediksi baru
            st.subheader("Prediksi Data Baru")
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
    else:
        st.info("Silakan unggah dataset terlebih dahulu untuk memulai.")

except Exception as e:
    st.error(f"Terjadi error: {e}")
