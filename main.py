from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import numpy as np
import re
import pickle

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[str, None]
    engine: Union[str, None]
    max_power: Union[str, None]
    torque: Union[str, None]
    seats: Union[float, None]


class Items(BaseModel):
    objects: List[Item]


def torque_feature(x):
    if x == x:
        x = x.lower()
        numbers = re.findall(r'\d+[\.\,]?\d*', x)
        torque = float(numbers[0])
        if x.find('kgm') != -1:
            torque *= 9.80665

        if len(numbers) == 1:
            rpm = np.nan
        elif len(numbers) == 2:
            rpm = int(numbers[1].replace(',', ''))
        else:
            num1, num2 = float(numbers[1].replace(',', '')), \
                         float(numbers[2].replace(',', ''))
            if num2 > num1 > 1000:
                rpm = (num1 + num2) / 2
            elif num1 < num2 and num1 < 1000:
                rpm = num2
            else:
                rpm = num1
        return pd.Series([torque, rpm])

    return pd.Series([np.nan, np.nan])


def feature_engineering_and_model(df):
    df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d+)').astype(float)
    df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
    df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d+)').astype(float)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(torque_feature)

    with open('model.pickle', 'rb') as handle:
        data_return = pickle.load(handle)

    df['was_nan'] = 0
    df.loc[df.isna().any(axis=1), 'was_nan'] = 1
    df = df.fillna(data_return['medians'])
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df['car'] = df['name'].apply(lambda x: x.split(' ')[0])
    df['year_2'] = df['year'] ** 2
    df['torque_log'] = np.log(df['torque'])
    df['max_power_log'] = np.log(df['max_power'])
    df['engine_log'] = np.log(df['engine'])
    df['km_driven_log'] = np.log(df['km_driven'])

    one_hot_enc = data_return['one_hot_enc']
    X_cat = df.drop(columns=['selling_price', 'name', 'year',
                             'torque', 'max_power', 'engine',
                             'km_driven']).reset_index(drop=True)

    arr_enc = one_hot_enc.transform(X_cat[['fuel', 'seller_type',
                                           'car', 'transmission',
                                           'owner']]).toarray()

    X_cat = X_cat.drop(columns=['fuel', 'seller_type', 'car',
                                'transmission', 'owner'])
    data_enc = pd.DataFrame(arr_enc,
                            columns=one_hot_enc.get_feature_names_out())

    X_cat = pd.concat([X_cat, data_enc], axis=1)
    X_cat = X_cat[data_return['columns']]

    X_scaled = data_return['scaler_x'].transform(X_cat)

    clf = data_return['model']
    y_pred = clf.best_estimator_.predict(X_scaled)
    ans = np.exp(data_return['scaler_y'].inverse_transform(y_pred))
    print(ans)
    return ans


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    answer = feature_engineering_and_model(df)[0][0]
    return answer


@app.post("/predict_items")
def upload_csv(csv_file: UploadFile = File(...)) -> List[float]:
    df = pd.read_csv(csv_file.file)
    print(df)
    answer = np.concatenate(feature_engineering_and_model(df), axis=0).tolist()
    return answer
