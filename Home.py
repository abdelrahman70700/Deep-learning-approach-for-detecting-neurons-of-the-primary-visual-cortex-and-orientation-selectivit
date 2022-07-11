import streamlit as st
from PIL import Image
import glob
def Home_Page():
	images = glob.glob("images/*.jpg")
	index= st.number_input('Index',0,10,0)

	# if st.button('Next'):
	#     index+=1


	# if st.button('Prev'):
	#     if index > 0:
	#         index = index -1

	image = Image.open(images[index])
	st.image(image, use_column_width=True)
	st.write("7mraaaaaaaaa")
Home_Page()