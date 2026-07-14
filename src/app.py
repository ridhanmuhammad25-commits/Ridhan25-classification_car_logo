import streamlit as st
import eda
import prediction


page = st.sidebar.selectbox('Pilih Halaman: ', ('Home', 'Prediction'))

if page == 'Home':
    eda.run()
else:
    prediction.run()