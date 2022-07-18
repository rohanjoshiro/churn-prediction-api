import numpy as np
import pandas as pd
import pathlib
from flask import Flask, request, jsonify, render_template
import Saaschurnpredictor
import os



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    paths = [str(x) for x in request.form.values()]
    final_features = paths
    dt=pd.read_csv(os.path.abspath(final_features[1]),index_col= False)
    dt_dict = (dt).to_dict(orient='records')
    df = pd.read_csv(os.path.abspath(final_features[0]),dtype=dt_dict[0])
    path=pathlib.Path(os.path.abspath(final_features[0])).parent.resolve()
    co=len(df.columns)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    mask = df['churn'].isnull()
    df2= (df[mask])
    df2 = df2.drop(columns='churn')
    total=len(df2)
    df=df.dropna(axis=0, how='all', subset=['churn'])


    df.churn = (df.churn == 'yes').astype(int)
    df3=Saaschurnpredictor.ChurnPrediction(df,df2)
    churning = (df3['churn score'] > 0.5).sum()



    df3.to_csv(os.path.join(path, 'prediction.csv'))
    
    return render_template('index1.html', prediction_text="you prediction file is stored in the same folder as of dataset csv file. Out of "+str(total)+" customers,"+str(churning)+" customers are expected to churn")


if __name__ == "__main__":
    app.run(debug=True)
    
