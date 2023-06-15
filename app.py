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

# Ubah tipe data kolom-kolom numerik menjadi float
data['latitude'] = data['latitude'].str.replace('.', '').astype(float)
data['longitude'] = data['longitude'].str.replace('.', '').astype(float)
data['brightness'] = data['brightness'].astype(float)
data['scan'] = data['scan'].astype(float)
data['track'] = data['track'].astype(float)
data['acq_time'] = data['acq_time'].astype(int)
data['confidence'] = data['confidence'].astype(int)
data['version'] = data['version'].astype(float)
data['bright_t31'] = data['bright_t31'].astype(float)
data['frp'] = data['frp'].astype(float)

# Ubah tipe data kolom acq_date menjadi datetime
data['acq_date'] = pd.to_datetime(data['acq_date'])

# Ubah tipe data kolom target 'type' menjadi integer
data['type'] = data['type'].astype(int)

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

# Tampilkan dataset
st.subheader("Data:")
st.write(data)

# Tampilkan hasil prediksi dan akurasi
st.subheader("Hasil Prediksi:")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(results)
st.subheader("Akurasi:")
st.write(accuracy)
