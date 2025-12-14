import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Vending Machine Classification",
    layout="wide"
)

# ===============================
# LOAD MODELS & PREPROCESSOR
# ===============================
rf_model = joblib.load("rf_vending_model.pkl")
nb_model = joblib.load("nb_vending_model.pkl")
le = joblib.load("label_encoder.pkl")
imputer = joblib.load("imputer.pkl")

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("vending_machine_sales.csv")

features = [
    "RPrice", "RQty", "MPrice", "MQty",
    "LineTotal", "TransTotal"
]

# ===============================
# PREPROCESSING (WAJIB)
# ===============================
X_raw = df[features]              # data masih ada NaN
X = imputer.transform(X_raw)      # data sudah bersih
y = le.transform(df["Category"])

# ===============================
# HEADER
# ===============================
st.title("🎯 Klasifikasi Produk Vending Machine")
st.caption("Perbandingan Algoritma Random Forest dan Naive Bayes")

tab1, tab2 = st.tabs(["🌲 Random Forest", "📊 Naive Bayes"])

# ======================================================
# TAB 1 — RANDOM FOREST
# ======================================================
with tab1:
    st.subheader("🌲 Random Forest Classifier")

    col1, col2 = st.columns(2)

    with col1:
        rf_rprice = st.number_input("RPrice", 0.0, key="rf_rprice")
        rf_rqty = st.number_input("RQty", 0, key="rf_rqty")
        rf_mprice = st.number_input("MPrice", 0.0, key="rf_mprice")

    with col2:
        rf_mqty = st.number_input("MQty", 0, key="rf_mqty")
        rf_linetotal = st.number_input("Line Total", 0.0, key="rf_linetotal")
        rf_transtotal = st.number_input("Transaction Total", 0.0, key="rf_transtotal")

    if st.button("Prediksi (Random Forest)"):
        rf_input_raw = np.array([[rf_rprice, rf_rqty, rf_mprice,
                                  rf_mqty, rf_linetotal, rf_transtotal]])
        rf_input = imputer.transform(rf_input_raw)
        rf_pred = rf_model.predict(rf_input)
        rf_category = le.inverse_transform(rf_pred)

        st.success(f"Hasil Prediksi: **{rf_category[0]}**")

    st.markdown("---")
    st.subheader("📊 Evaluasi Random Forest")

    y_pred_rf = rf_model.predict(X)
    acc_rf = accuracy_score(y, y_pred_rf)
    st.metric("Accuracy", f"{acc_rf:.2f}")

    cm_rf = confusion_matrix(y, y_pred_rf)
    fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_rf, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax_rf
    )
    ax_rf.set_title("Confusion Matrix - Random Forest")
    ax_rf.set_xlabel("Predicted")
    ax_rf.set_ylabel("Actual")
    st.pyplot(fig_rf)

# ======================================================
# TAB 2 — NAIVE BAYES
# ======================================================
with tab2:
    st.subheader("📊 Naive Bayes Classifier")

    col3, col4 = st.columns(2)

    with col3:
        nb_rprice = st.number_input("RPrice", 0.0, key="nb_rprice")
        nb_rqty = st.number_input("RQty", 0, key="nb_rqty")
        nb_mprice = st.number_input("MPrice", 0.0, key="nb_mprice")

    with col4:
        nb_mqty = st.number_input("MQty", 0, key="nb_mqty")
        nb_linetotal = st.number_input("Line Total", 0.0, key="nb_linetotal")
        nb_transtotal = st.number_input("Transaction Total", 0.0, key="nb_transtotal")

    if st.button("Prediksi (Naive Bayes)"):
        nb_input_raw = np.array([[nb_rprice, nb_rqty, nb_mprice,
                                  nb_mqty, nb_linetotal, nb_transtotal]])
        nb_input = imputer.transform(nb_input_raw)
        nb_pred = nb_model.predict(nb_input)
        nb_category = le.inverse_transform(nb_pred)

        st.success(f"Hasil Prediksi: **{nb_category[0]}**")

    st.markdown("---")
    st.subheader("📊 Evaluasi Naive Bayes")

    y_pred_nb = nb_model.predict(X)
    acc_nb = accuracy_score(y, y_pred_nb)
    st.metric("Accuracy", f"{acc_nb:.2f}")

    cm_nb = confusion_matrix(y, y_pred_nb)
    fig_nb, ax_nb = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_nb, annot=True, fmt="d", cmap="Greens",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax_nb
    )
    ax_nb.set_title("Confusion Matrix - Naive Bayes")
    ax_nb.set_xlabel("Predicted")
    ax_nb.set_ylabel("Actual")
    st.pyplot(fig_nb)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Data Mining | Classification | Random Forest vs Naive Bayes")
