import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Tampilkan judul aplikasi
st.title("Prediksi Kebakaran Hutan dan Lahan")

# Ubah URL dengan URL file dataset yang sesuai di repositori GitHub Anda
dataset_url = "https://raw.githubusercontent.com/badew2912/tugasds/main/modis_2018-2022_Indonesia.csv"

# Baca dataset
data = pd.read_csv(dataset_url)

# Tampilkan dataset
st.subheader("Data:")
st.write(data)

# Bagi dataset menjadi fitur dan target
X = data.drop('type', axis=1)
y = data['type']

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Decision Tree Classifier
clf = DecisionTreeClassifier()

# Latih model
clf.fit(X_train, y_train)

# Prediksi data uji
y_pred = clf.predict(X_test)

# Hitung akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Tampilkan hasil prediksi dan akurasi
st.subheader("Hasil Prediksi:")
st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
st.subheader("Akurasi:")
st.write(accuracy)
