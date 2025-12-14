import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Vending Machine Data Mining",
    layout="wide"
)

# ===============================
# LOAD FILES
# ===============================
clf_model = joblib.load("rf_vending_model.pkl")      # klasifikasi
reg_model = joblib.load("rf_regression_model.pkl")   # regresi
le = joblib.load("label_encoder.pkl")

df = pd.read_csv("vending_machine_sales.csv")

features_clf = [
    "RPrice", "RQty", "MPrice", "MQty",
    "LineTotal", "TransTotal"
]

features_reg = [
    "RPrice", "RQty", "MPrice", "MQty",
    "LineTotal"
]

X = df[features_clf]
y = le.transform(df["Category"])

# ===============================
# UI HEADER
# ===============================
st.title("🎯 Klasifikasi & Regresi Produk Vending Machine")
st.caption("Algoritma: Random Forest (Ensemble Method)")

tab1, tab2 = st.tabs(["📊 Klasifikasi", "📈 Regresi"])

# ======================================================
# TAB 1 — KLASIFIKASI
# ======================================================
with tab1:
    st.subheader("🔮 Prediksi Kategori Produk")

    col1, col2 = st.columns(2)

    with col1:
        rprice = st.number_input("RPrice", 0.0, key="c_rprice")
        rqty = st.number_input("RQty", 0, key="c_rqty")
        mprice = st.number_input("MPrice", 0.0, key="c_mprice")

    with col2:
        mqty = st.number_input("MQty", 0, key="c_mqty")
        linetotal = st.number_input("Line Total", 0.0, key="c_linetotal")
        transtotal = st.number_input("Transaction Total", 0.0, key="c_transtotal")

    if st.button("Prediksi Kategori"):
        data = np.array([[rprice, rqty, mprice, mqty, linetotal, transtotal]])
        pred = clf_model.predict(data)
        category = le.inverse_transform(pred)
        st.success(f"Kategori Produk: **{category[0]}**")

    st.markdown("---")
    st.subheader("📊 Evaluasi Model Klasifikasi")

    y_pred = clf_model.predict(X)
    acc = accuracy_score(y, y_pred)

    st.metric("Accuracy", f"{acc:.2f}")

    cm = confusion_matrix(y, y_pred)
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

# ======================================================
# TAB 2 — REGRESI
# ======================================================
with tab2:
    st.subheader("📈 Prediksi Total Transaksi (Regresi)")
    st.write("Target Regresi: **TransTotal**")
    st.write("Algoritma: **Random Forest Regressor**")

    col3, col4 = st.columns(2)

    with col3:
        rprice_r = st.number_input("RPrice", 0.0, key="r_rprice")
        rqty_r = st.number_input("RQty", 0, key="r_rqty")
        mprice_r = st.number_input("MPrice", 0.0, key="r_mprice")

    with col4:
        mqty_r = st.number_input("MQty", 0, key="r_mqty")
        linetotal_r = st.number_input("Line Total", 0.0, key="r_linetotal")

    if st.button("Prediksi Total Transaksi"):
        reg_input = np.array([[rprice_r, rqty_r, mprice_r, mqty_r, linetotal_r]])
        pred_total = reg_model.predict(reg_input)
        st.success(f"Prediksi Total Transaksi: **{pred_total[0]:,.2f}**")

    st.markdown("---")
    st.info(
        "Regresi digunakan untuk memprediksi nilai numerik total transaksi, "
        "sementara klasifikasi digunakan untuk menentukan kategori produk."
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Data Mining | Classification & Regression | Streamlit Cloud")
