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
st.set_page_config(page_title="Natural Neural Vision Algorithms", page_icon=":house:", layout="wide")


with st.sidebar:
    choose = option_menu("Natural Neural Vision Algorithms", ["About", "DL Model","Data Analysis", "Contact Us"],
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
from about_page import about
from contact_page import contact
from edit_page import run
from dividing import final_processing


if choose == "About":
    obj1=about()
    obj1.about_page()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")
elif choose == "DL Model":
    emp=run()
    emp.start()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")

elif choose=="Data Analysis":
    analyse=final_processing()
    analyse.analyse()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")
elif choose == "Contact Us":
    contact_page=contact()
    contact_page.contact_page()
    try:
        shutil.rmtree("images")
    except FileNotFoundError:
        print("")
