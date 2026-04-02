import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HW3_BaseFunc import load_data, process_time, latlon_to_meters, prepare_speed


def prepare_navigation(df):
    # bearing в радианы
    df['Bearing'] = pd.to_numeric(df[' Bearing'], errors='coerce')
    df['bearing_rad'] = np.radians(df[' Bearing'])

    # используем уже посчитанный y (широта!)
    df['pos_meas'] = df['y']
    return df

class KalmanPosition1D:
    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self, v, theta, dt):#Фаза движения
        if np.isnan(v) or np.isnan(theta):
            return
        dx = v * np.cos(theta) * dt
        self.x = self.x + dx

        self.P = self.P + self.Q * dt

    def update(self, z):#Фаза сканирования
        if np.isnan(z):
            return

        K = self.P / (self.P + self.R)

        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P

def run_kalman_position(df):
    kf = KalmanPosition1D(
        x0=0,
        P0=100,  # Начальный шум
        Q=5,     # шум модели (движение)
        R=20     # шум GPS позиции
    )

    x_est = []
    P_est = []

    t = df['t'].values
    v = df['v_gps'].values
    theta = df['bearing_rad'].values
    z = df['pos_meas'].values

    for i in range(len(df)):
        dt = 0 if i == 0 else t[i] - t[i-1]

        # --- MOTION ---
        kf.predict(v[i], theta[i], dt)
        # --- SENSE ---
        kf.update(z[i])

        x_est.append(kf.x)
        P_est.append(kf.P)
    df['x_kalman_pos'] = x_est
    df['P_pos'] = P_est
    return df

def plot_position_kalman(df):

    plt.figure(figsize=(12, 6))

    # измерения
    plt.plot(df['t'], df['pos_meas'], label='GPS position', alpha=0.5)

    # калман
    plt.plot(df['t'], df['x_kalman_pos'], label='Kalman position', linewidth=2)

    # uncertainty
    sigma = np.sqrt(df['P_pos'])
    plt.fill_between(
        df['t'],
        df['x_kalman_pos'] - 3*sigma,
        df['x_kalman_pos'] + 3*sigma,
        alpha=0.2,
        label='3σ'
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Kalman Filter (Position along latitude)")
    plt.legend()
    plt.grid()

    plt.show()

df = load_data()
process_time(df)
latlon_to_meters(df)
prepare_speed(df)
prepare_navigation(df)
run_kalman_position(df)
plot_position_kalman(df)