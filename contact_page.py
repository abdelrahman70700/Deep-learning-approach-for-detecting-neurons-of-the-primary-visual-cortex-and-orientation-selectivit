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
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



#-------------------------------------------------------------------------------------------------------------------
def contact_page():
    logo = Image.open('index.png')
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Get In Touch With Us!</p>', unsafe_allow_html=True)
        
    with col2:               # To display brand logo
        st.image(logo,  width=130)
    local_css("style/style.css")
    contact_form = """
    <form action="https://formsubmit.co/abdohassan7001@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    
    st.markdown(contact_form, unsafe_allow_html=True)
    
    # st.markdown(""" <style> .font {
    # font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    # </style> """, unsafe_allow_html=True)
    # st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    # with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    #     #st.write('Please help us improve!')
    #     Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
    #     Email=st.text_input(label='Please Enter Email') #Collect user feedback
    #     Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
    #     submitted = st.form_submit_button('Submit')
    #     if submitted:
    #         st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')    

#-------------------------------------------------------------------------------------------

