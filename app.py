import matplotlib.pyplot as plt
import statistics
import math
import scipy.io
from streamlit_option_menu import option_menu
import shutil
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
#---------------------------------------------------------------------------------------------
st.set_page_config(page_title="Computational Brain Anatomy", page_icon=":house:", layout="wide")


with st.sidebar:
    choose = option_menu("Computational Brain Anatomy", ["About", "DL Model","Data Analysis", "Contact Us"],
                         icons=['house-fill', 'camera-reels-fill','bar-chart-line-fill','person-fill'],

                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},

    },
    )


#---------------------------------------------------------------------------------------------------
from about_page import about_page
from contact_page import contact_page
from edit_page import edit_page
from dividing import analyse


if choose == "About":
    about_page()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")
elif choose == "DL Model":
    edit_page()

elif choose=="Data Analysis":
    analyse()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")
elif choose == "Contact Us":
    contact_page()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")
