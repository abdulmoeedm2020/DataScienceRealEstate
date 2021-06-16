import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.write("""# Real Estate Price prediction""")
X = pd.read_csv("df12.csv")

form = st.form(key = 'my_form')
location = form.selectbox('location',('Lahore','Islamabad','Bahria Town Karachi'))
baths = form.slider('baths',1,10,1)
area_sqft = form.slider('area_sqft',10000,100000,10000)
bedrooms = form.slider('Bed Room',1,30,3)
submit = form.form_submit_button(label='Submit')

if submit:

	loc_index = np.where(X.columns==location)[0][0]
	print(loc_index)
	x = np.zeros(len(X.columns))
	print(x)
	x[0] = baths
	x[1] = area_sqft
	x[2] = bedrooms
	if loc_index >= 0:
		x[loc_index] = 1

	load_clf = pickle.load(open('PPP.pickle', 'rb'))
	prediction = load_clf.predict([x])[0]
	st.write(prediction)
