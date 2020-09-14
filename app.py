from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('car_rf_model')

st.set_option('deprecation.showfileUploaderEncoding', False)

from PIL import Image
image_car = Image.open('car.jpg')
st.title("HOW MUCH COST A CAR ?")
st.image(image_car,use_column_width=True)
add_selectbox = st.sidebar.selectbox("How do you want to predict ?",("Online", "CSV File"))
st.sidebar.info("This program tries to predict the value of a car depending on several features")
st.sidebar.success('https://www.pycaret.org')


def predict(model, input_df):
    predictions= model.predict(input_df)
    return np.round(predictions, 2)



def run():

    if add_selectbox == 'Online':
        brand=st.selectbox('Choose your Brand' ,['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'buick', 'mercury',
       'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche',
       'renault', 'saab', 'subaru', 'toyota', 
        'volkswagen', 'volvo'])
        enginesize =st.number_input('Engine Size',min_value=50, max_value=500, value=70)
        horsepower= st.number_input('Horsepower', min_value=40, max_value=300, value=100)
        highwaympg = st.number_input("Highway MPG", min_value=15, max_value=60, value=30)
        curbweight = st.number_input('Curbweight', min_value=1400, max_value=5000, value=2000)
        compressionratio= st.number_input('Compression Ratio',  min_value=5, max_value=30, value=7)
    #     promotion_last_5years = st.number_input('Promotion 5 dernières années',  min_value=0, max_value=50, value=0)
    #     department= st.selectbox('Département', ['comptabilite', 'technique', 'IT', 'support', 'R&D', 'ventes',
    #    'management', 'marketing', 'product_mng', 'rh'])
    #     salary = st.selectbox('Salaire', ['bas', 'normal','eleve'])
        output=""
        input_dict={'brand':brand,'enginesize':enginesize,'compressionratio':compressionratio,
        'horsepower':horsepower,'highwaympg': highwaympg,'curbweight':curbweight}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('You might pay around {} $'.format(output))
    if add_selectbox == 'CSV File':
        file_upload = st.file_uploader("Please upload your file ", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

run()


