import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ===============================
# LOAD MODEL & DATA
# ===============================
model = joblib.load('rf_vending_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load dataset (untuk visualisasi)
df = pd.read_csv('vending_machine_sales.csv')

features = [
    'RPrice', 'RQty', 'MPrice', 'MQty',
    'LineTotal', 'TransTotal'
]

X = df[features]
y = le.transform(df['Category'])

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Vending Machine Classification", layout="wide")

st.title("🎯 Klasifikasi Produk Vending Machine")
st.markdown("**Algoritma:** Random Forest (Ensemble Method)")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("📝 Input Data Transaksi")

RPrice = st.sidebar.number_input("RPrice", 0.0)
RQty = st.sidebar.number_input("RQty", 0)
MPrice = st.sidebar.number_input("MPrice", 0.0)
MQty = st.sidebar.number_input("MQty", 0)
LineTotal = st.sidebar.number_input("Line Total", 0.0)
TransTotal = st.sidebar.number_input("Transaction Total", 0.0)

# ===============================
# PREDIKSI
# ===============================
st.subheader("🔮 Prediksi Kategori Produk")

if st.button("Prediksi"):
    input_data = np.array([[RPrice, RQty, MPrice, MQty, LineTotal, TransTotal]])
    pred = model.predict(input_data)
    category = le.inverse_transform(pred)

    st.success(f"Kategori Produk: **{category[0]}**")

# ===============================
# VISUALISASI MODEL
# ===============================
st.subheader("📊 Evaluasi & Visualisasi Model")

# ===== Confusion Matrix =====
st.markdown("### 🔹 Confusion Matrix")

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")

st.pyplot(fig_cm)

# ===== Accuracy =====
acc = accuracy_score(y, y_pred)
st.markdown(f"**Akurasi Model:** `{acc:.2f}`")

# ===== Feature Importance =====
st.markdown("### 🔹 Feature Importance")

importance = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=True)

fig_fi, ax_fi = plt.subplots(figsize=(6,4))
ax_fi.barh(feat_imp['Feature'], feat_imp['Importance'])
ax_fi.set_title("Feature Importance - Random Forest")

st.pyplot(fig_fi)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Data Mining | Classification | Ensemble Method (Random Forest)")
