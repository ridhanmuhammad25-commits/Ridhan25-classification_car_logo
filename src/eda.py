import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "df_counts.csv")


def run():

    st.set_page_config(
        page_title="Car Classification",
        page_icon="🚗",
        layout="centered"
    )

    st.markdown(
    "<h1 style='text-align: center;'>🚗 Car Classification</h1>",
    unsafe_allow_html=True
    )
    img = Image.open(os.path.join('src','title.jpg'))
    st.image(img, caption = '')
    st.subheader("Data set Overview")
    st.markdown(
        """
        Welcome to the **Car Classification** application.

        This application can classify car into **5 brands**:
        - 🚘 BMW
        - ⭐ Mercedes
        - 🚙 Toyota
        - 🏁 Bentley

        Below is the distribution of training images used for each class.
        """
    )

    # ===============================
    # Load CSV
    # ===============================
    df = pd.read_csv(CSV_PATH)

    # ===============================
    # Statistik
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Classes", len(df))

    with col2:
        st.metric("Total Images", int(df["jumlah_gambar"].sum()))

    st.divider()
    st.subheader("📋 Dataset Summary")
    st.dataframe(df, use_container_width=True)
    st.subheader("📊 Image Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        data=df,
        x="class",
        y="jumlah_gambar",
        hue="class",
        palette="viridis",
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Car Brand")
    ax.set_ylabel("Number of Images")
    ax.set_title("Training Images per Class")

    # Menampilkan jumlah di atas bar
    for container in ax.containers:
        ax.bar_label(container)

    st.pyplot(fig)

    st.divider()

    st.info(
        "💡 This dataset is used to train a deep learning model for recognizing car logos from uploaded images."
    )


if __name__ == "__main__":
    run()