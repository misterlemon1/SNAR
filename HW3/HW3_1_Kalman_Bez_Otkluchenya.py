import numpy as np
from HW3_BaseFunc import load_data, process_time, latlon_to_meters, prepare_speed, simulate_dropout_by_time
import matplotlib.pyplot as plt
import pandas as pd

class Kalman1D:
    def __init__(self, x0, P0, Q, R_gps, R_obd):
        self.x = x0      # оценка скорости
        self.P = P0      # дисперсия
        self.Q = Q       # шум модели
        self.R_gps = R_gps
        self.R_obd = R_obd

    def predict(self, dt):
        # модель: V = V (ничего не меняется)
        # но дисперсия растёт
        self.P = self.P + self.Q * dt

    def update(self, z, R):
        # если нет измерения — ничего не делаем
        if np.isnan(z):
            return

        K = self.P / (self.P + R)

        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P



def run_kalman(df):
    # инициализация
    kf = Kalman1D(
        x0=0,        # начальная скорость
        P0=10,       # начальная неопределённость
        Q=1,         # шум модели (настроечный!)
        R_gps=5,     # шум GPS
        R_obd=1      # шум OBD (точнее)
    )

    v_est = []
    P_est = []

    t = df['t'].values
    v_gps = df['v_gps'].values
    v_obd = df['v_obd'].values

    for i in range(len(df)):
        if i == 0:
            dt = 0
        else:
            dt = t[i] - t[i-1]

        # --- PREDICT ---
        kf.predict(dt)

        # --- UPDATE OBD ---
        if not np.isnan(v_obd[i]):
            kf.update(v_obd[i], kf.R_obd)

        # --- UPDATE GPS ---
        if not np.isnan(v_gps[i]):
            #важно: GPS не обновлять, если это дубликат
            if i == 0 or v_gps[i] != v_gps[i-1]:
                kf.update(v_gps[i], kf.R_gps)

        v_est.append(kf.x)
        P_est.append(kf.P)

    df['v_kalman'] = v_est
    df['P'] = P_est

    return df

def add_boundaries(ax, df, start_time, end_time, color):
    mask = (
        (pd.to_datetime(df[' Device Time']).dt.time >= pd.to_datetime(start_time).time()) &
        (pd.to_datetime(df[' Device Time']).dt.time <= pd.to_datetime(end_time).time())
    )

    filtered = df[mask]
    if filtered.empty:
        return

    first_row = filtered.iloc[0]
    last_row = filtered.iloc[-1]

    ax.plot(
        [first_row['t'], first_row['t']],
        [
            first_row['v_kalman'] + 10 * np.sqrt(first_row["P"]),
            first_row['v_kalman'] - 10 * np.sqrt(first_row["P"])
        ],
        "--",
        color=color
    )

    ax.plot(
        [last_row['t'], last_row['t']],
        [
            last_row['v_kalman'] + 10 * np.sqrt(last_row["P"]),
            last_row['v_kalman'] - 10 * np.sqrt(last_row["P"])
        ],
        "--",
        color=color
    )

def draw_kalman(ax, df,boundaries=False):
    ax.plot(df['t'], df['v_obd'], label='OBD', alpha=0.5)
    ax.plot(df['t'], df['v_gps'], label='GPS', alpha=0.5)
    ax.plot(df['t'], df['v_kalman'], label='Kalman', linewidth=2)

    sigma = np.sqrt(df['P'])
    ax.fill_between(
        df['t'],
        df['v_kalman'] - 3 * sigma,
        df['v_kalman'] + 3 * sigma,
        alpha=0.2,
        label='3σ'
    )
    if boundaries:
        add_boundaries(ax, df, '17:25', '17:35', 'r')
        add_boundaries(ax, df, '17:40', '17:50', 'k')
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Kalman Filter (Speed)")
    ax.legend()

def plot_kalman(df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    draw_kalman(axes[0], df)
    draw_kalman(axes[1], df,True)

    plt.tight_layout()
    plt.show()



df = load_data()
df = process_time(df)
df = latlon_to_meters(df)
df = prepare_speed(df)
df = run_kalman(df)

plot_kalman(df)
