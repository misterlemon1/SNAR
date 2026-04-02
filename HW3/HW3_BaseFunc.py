import numpy as np
import pandas as pd

def load_data(): #Просто загрузка данных
    df = pd.read_csv("data1.csv")
    return df

def process_time(df): #Преобразование времени, чтобы показывало сколько прошло со старта
    df[' Device Time'] = pd.to_datetime(df[' Device Time'])
    t0 = df[' Device Time'].iloc[0]
    df['t'] = (df[' Device Time'] - t0).dt.total_seconds()
    return df

def simulate_dropout_by_time(df):
    df = df.copy()

    # убедимся, что время — datetime
    df[' Device Time'] = pd.to_datetime(df[' Device Time'])

    # --- GPS пропадает 17:25–17:35 ---
    gps_mask = (
        (df[' Device Time'].dt.time >= pd.to_datetime("17:25").time()) &
        (df[' Device Time'].dt.time <= pd.to_datetime("17:35").time())
    )
    df.loc[gps_mask, 'v_gps'] = np.nan

    # --- OBD пропадает 17:40–17:50 ---
    obd_mask = (
        (df[' Device Time'].dt.time >= pd.to_datetime("17:40").time()) &
        (df[' Device Time'].dt.time <= pd.to_datetime("17:50").time())
    )
    df.loc[obd_mask, 'v_obd'] = np.nan

    return df

def latlon_to_meters(df): #Перевод широты и долготы
    R = 6371000  # радиус земли в метрах
    lat = np.radians(df[' Latitude'].values) #Перевод в радианы
    lon = np.radians(df[' Longitude'].values)
    lat0 = lat[0] #Начальная точка
    lon0 = lon[0]
    x = (lon - lon0) * R * np.cos(lat0) #Сам перевод
    y = (lat - lat0) * R
    df['x'] = x #Забиваем в таблицу для кайфа
    df['y'] = y
    return df

def prepare_speed(df): #Для кайфа для удобства
    df['Speed (GPS)(km/h)'] = pd.to_numeric(df['Speed (GPS)(km/h)'], errors='coerce') # перевод в float
    df['Speed (OBD)(km/h)'] = pd.to_numeric(df['Speed (OBD)(km/h)'], errors='coerce')

    df['v_gps'] = df['Speed (GPS)(km/h)'] / 3.6 #Перевод в м/c
    df['v_obd'] = df['Speed (OBD)(km/h)'] / 3.6
    return df

if __name__=="__main__":
    df=load_data()
    df=process_time(df)
    df = latlon_to_meters(df)
    df = prepare_speed(df)
    print(df.head(10))
    print("Поля:")
    print([*df])