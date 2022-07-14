import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import time
import hydralit_components as hc
class about:
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    def about_page(self):
        logo = Image.open('index.png')
        lottie_coding = about.load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_33asonmr.json")
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Natural Neural Vision Algorithm</p>', unsafe_allow_html=True)    
        with col2:               # To display brand log
            st.image(logo, width=130 )
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("What WE do")
            st.write("##")
            st.write(
                """
                On Our project We will :
                - Describe the features of images important for vision.
                - Define dendrites, soma, axon, synapses, and action potential. 
                - Identify the anatomy of the visual system. 
                - Load and analyze visual neuroscience data using Deep learning.
                - Plot neuroscience data using PYTHON. 
                - Using ML and DL to predict the place of visual system
                """
            )
        st.write("---")
        with right_column:
            st_lottie(lottie_coding, height=500, key="coding")
