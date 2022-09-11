from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


model = joblib.load(r"D:\Harshit_Gadhiya_Step_by_stepvideo\Self_made_project\Item_outlet_sales_price_prediction\model.pkl")

df = pd.read_csv(r"D:\Harshit_Gadhiya_Step_by_stepvideo\Self_made_project\Item_outlet_sales_price_prediction\Train.csv")

num_cols = df.select_dtypes(["int64", "float64"]).keys()
cat_cols = df.select_dtypes("O").keys()

for var in cat_cols:
    df[var].fillna(df[var].mode()[0], inplace=True)

for var in num_cols:
    df[var].fillna(df[var].mean(), inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

for var in cat_cols:
    le.fit(df1[var])
    df1[var]=le.transform(df1[var])

di = {}

for var1 in cat_cols:
    for var2,var3 in zip(df[var1],df1[var1]):
        if var2 not in di.keys():
            di[var2]=var3

X=df.drop("Item_Outlet_Sales", axis=1)
y=df["Item_Outlet_Sales"]


def house_price_prediction(model, Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Item_Identifier, 
                            Item_Fat_Content, Outlet_Size, Item_Type, Outlet_Identifier, Outlet_Location_Type, Outlet_Type):
    x=np.zeros(len(X.columns))
    x[0]= Item_Weight
    x[1]=Item_Visibility
    x[2]=Item_MRP
    x[3]=Outlet_Establishment_Year
    x[4] = di[Item_Identifier]
    x[5] = di[Item_Fat_Content]
    x[6] = di[Outlet_Size]
    x[7] = di[Item_Type]
    x[8] = di[Outlet_Identifier]
    x[9] = di[Outlet_Location_Type]
    x[10] = di[Outlet_Type]
        
    
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Item_Weight = request.form["Item_Weight"]
    Item_Visibility = request.form["Item_Visibility"]
    Item_MRP = request.form["Item_MRP"]
    Outlet_Establishment_Year = request.form["Outlet_Establishment_Year"]
    Item_Identifier = request.form["Item_Identifier"]
    Item_Fat_Content = request.form["Item_Fat_Content"]
    Outlet_Size = request.form["Outlet_Size"]
    Item_Type = request.form["Item_Type"]
    Outlet_Identifier = request.form["Outlet_Identifier"]
    Outlet_Location_Type = request.form["Outlet_Location_Type"]
    Outlet_Type = request.form["Outlet_Type"]
    
    predicated_price1 =house_price_prediction(model, Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Item_Identifier, 
                                            Item_Fat_Content, Outlet_Size, Item_Type, Outlet_Identifier, Outlet_Location_Type, Outlet_Type)
    predicated_price = round(predicated_price1, 2)

    return render_template("index.html", prediction_text="Predicated price of Item-outlet sales is {} RS".format(predicated_price))

if __name__ == "__main__":
    app.run() 