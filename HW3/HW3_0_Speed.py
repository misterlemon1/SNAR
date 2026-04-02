import numpy as np
from HW3_BaseFunc import prepare_speed, load_data, process_time, latlon_to_meters
import matplotlib.pyplot as plt

df = load_data()
process_time(df)
latlon_to_meters(df)
prepare_speed(df) #Подготовка

def compute_speed_from_xy(df):
    dx = np.diff(df['x'])
    dy = np.diff(df['y'])
    dt = np.diff(df['t'])

    dt[dt == 0] = np.nan #защита от деления на 0

    v = np.sqrt(dx**2 + dy**2) / dt

    df['v_xy'] = np.insert(v, 0, 0) # выравниваем размер
    #Это необходимо, так как после np.diff у нас одно значение пропадает
    # и кол-во индексов не совпадает
    return df


def plot_speeds(df):
    plt.figure(figsize=(12, 6))

    plt.plot(df['t'], df['v_obd'], label='OBD', linewidth=1.5)
    plt.plot(df['t'], df['v_gps'], label='GPS', alpha=0.7)
    plt.plot(df['t'], df['v_xy'], label='From XY', linestyle='--')

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed comparison")

    plt.legend()
    plt.grid()
    plt.show()

compute_speed_from_xy(df)
plot_speeds(df)