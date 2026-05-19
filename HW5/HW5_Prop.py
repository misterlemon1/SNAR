import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="data1.csv"):
    return pd.read_csv(path)

def process_time(df):

    df = df.copy()

    df[' Device Time'] = pd.to_datetime(
        df[' Device Time']
    )

    t0 = df[' Device Time'].iloc[0]

    df['t'] = (
        df[' Device Time'] - t0
    ).dt.total_seconds()

    return df

def latlon_to_meters(df):

    df = df.copy()

    R = 6371000

    lat = np.radians(
        df[' Latitude'].values
    )

    lon = np.radians(
        df[' Longitude'].values
    )

    lat0 = lat[0]
    lon0 = lon[0]

    df['x'] = (
        (lon - lon0)
        * R
        * np.cos(lat0)
    )

    df['y'] = (
        (lat - lat0)
        * R
    )

    return df

def prepare_speed(df):

    df = df.copy()

    df['Speed (GPS)(km/h)'] = pd.to_numeric(
        df['Speed (GPS)(km/h)'],
        errors='coerce'
    )

    df['Speed (OBD)(km/h)'] = pd.to_numeric(
        df['Speed (OBD)(km/h)'],
        errors='coerce'
    )

    df['v_gps'] = (
        df['Speed (GPS)(km/h)'] / 3.6
    )

    df['v_obd'] = (
        df['Speed (OBD)(km/h)'] / 3.6
    )

    return df

def prepare_initial_yaw(df):

    df = df.copy()

    dx = np.gradient(df['x'])
    dy = np.gradient(df['y'])

    yaw = np.arctan2(dy, dx)

    yaw = np.unwrap(yaw)

    yaw = (
        pd.Series(yaw)
        .rolling(
            window=25,
            center=True,
            min_periods=1
        )
        .mean()
    )

    df['yaw_init'] = yaw.values

    return df

def apply_dropout(
    df,
    mode=None,
    start_time="17:25",
    end_time="17:26"
):

    df = df.copy()

    time_mask = (
        (
            df[' Device Time'].dt.time >=
            pd.to_datetime(start_time).time()
        )
        &
        (
            df[' Device Time'].dt.time <=
            pd.to_datetime(end_time).time()
        )
    )

    # GNSS DROPOUT

    if mode == "GNSS":

        df.loc[time_mask, 'x'] = np.nan
        df.loc[time_mask, 'y'] = np.nan

    # OBD DROPOUT

    if mode == "OBD":

        df.loc[time_mask, 'v_obd'] = np.nan

    return df, time_mask

def normalize_angle(angle):

    return (
        (angle + np.pi)
        % (2*np.pi)
        - np.pi
    )

class EKF:

    def __init__(self, dt):

        self.dt = dt

        # [x, y, v, yaw]

        self.x = np.zeros((4,1))

        self.P = np.eye(4) * 5

        self.Q = np.diag([
            0.2,
            0.2,
            0.5,
            0.005
        ])

        self.R_gps = np.diag([
            4.0,
            4.0
        ])

        self.R_obd = np.array([
            [0.5]
        ])

    def predict(self):

        x = self.x[0,0]
        y = self.x[1,0]
        v = self.x[2,0]
        yaw = self.x[3,0]

        dt = self.dt

        x_new = (
            x +
            v * np.cos(yaw) * dt
        )

        y_new = (
            y +
            v * np.sin(yaw) * dt
        )

        self.x = np.array([
            [x_new],
            [y_new],
            [v],
            [yaw]
        ])

        F = np.array([
            [
                1,
                0,
                np.cos(yaw)*dt,
                -v*np.sin(yaw)*dt
            ],
            [
                0,
                1,
                np.sin(yaw)*dt,
                v*np.cos(yaw)*dt
            ],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.P = (
            F @ self.P @ F.T
            + self.Q
        )


    def update_gps(self, z):

        H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        z = z.reshape((2,1))

        innovation = (
            z - H @ self.x
        )

        S = (
            H @ self.P @ H.T
            + self.R_gps
        )

        K = (
            self.P @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + K @ innovation
        )

        I = np.eye(4)

        self.P = (
            I - K @ H
        ) @ self.P

    def update_obd(self, speed):

        H = np.array([
            [0,0,1,0]
        ])

        z = np.array([
            [speed]
        ])

        innovation = (
            z - H @ self.x
        )

        S = (
            H @ self.P @ H.T
            + self.R_obd
        )

        K = (
            self.P @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + K @ innovation
        )

        I = np.eye(4)

        self.P = (
            I - K @ H
        ) @ self.P

def update_yaw_from_motion(
    ekf,
    prev_x,
    prev_y,
    dt
):

    dx = (
        ekf.x[0,0]
        - prev_x
    )

    dy = (
        ekf.x[1,0]
        - prev_y
    )

    dist = np.hypot(dx, dy)

    v = ekf.x[2,0]

    min_dist = max(
        0.5,
        v * dt * 0.3
    )

    if dist > min_dist:

        new_yaw = np.arctan2(
            dy,
            dx
        )

        current_yaw = ekf.x[3,0]

        dyaw = normalize_angle(
            new_yaw - current_yaw
        )

        max_yaw_step = np.radians(10)

        dyaw = np.clip(
            dyaw,
            -max_yaw_step,
            max_yaw_step
        )

        alpha = 0.99

        ekf.x[3,0] = normalize_angle(
            current_yaw +
            (1 - alpha) * dyaw
        )

    return ekf.x[3,0]

def run_filter(df):

    dt = np.median(
        np.diff(df['t'])
    )

    ekf = EKF(dt)

    states = []

    ekf.x[0,0] = df['x'].iloc[0]
    ekf.x[1,0] = df['y'].iloc[0]

    if not np.isnan(df['v_obd'].iloc[0]):
        ekf.x[2,0] = df['v_obd'].iloc[0]
    else:
        ekf.x[2,0] = df['v_gps'].iloc[0]

    ekf.x[3,0] = df['yaw_init'].iloc[0]

    prev_x = ekf.x[0,0]
    prev_y = ekf.x[1,0]

    for i in range(len(df)):

        if i > 0:

            ekf.dt = (
                df['t'].iloc[i]
                - df['t'].iloc[i-1]
            )

        ekf.predict()

        # GPS UPDATE

        gps_available = (
            not np.isnan(df['x'].iloc[i])
            and
            not np.isnan(df['y'].iloc[i])
        )

        if gps_available:

            z = np.array([
                df['x'].iloc[i],
                df['y'].iloc[i]
            ])

            ekf.update_gps(z)

            update_yaw_from_motion(
                ekf,
                prev_x,
                prev_y,
                ekf.dt
            )

            prev_x = ekf.x[0,0]
            prev_y = ekf.x[1,0]

        # OBD UPDATE

        if not np.isnan(df['v_obd'].iloc[i]):

            ekf.update_obd(
                df['v_obd'].iloc[i]
            )

        states.append(
            ekf.x.flatten()
        )

    return np.array(states)

def plot_test(
    df,
    states,
    dropout_mask,
    title
):

    fig = plt.figure(figsize=(16,7))

    # TRAJECTORY

    ax1 = plt.subplot2grid(
        (2,2),
        (0,0),
        rowspan=2
    )

    # GPS trajectory

    ax1.plot(
        df['x'],
        df['y'],
        label='GPS',
        alpha=0.4
    )

    # EKF trajectory

    ax1.plot(
        states[:,0],
        states[:,1],
        label='EKF',
        linewidth=2
    )

    # DROPOUT SEGMENT

    drop_idx = np.where(dropout_mask)[0]

    ax1.plot(
        states[drop_idx, 0],
        states[drop_idx, 1],
        color='red',
        linewidth=6,
        alpha=0.25,
        solid_capstyle='round',
        label='Dropout interval'
    )

    ax1.set_title(
        f'Trajectory ({title})'
    )

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    ax1.axis('equal')

    ax1.grid()
    ax1.legend()

    # SPEED

    ax2 = plt.subplot2grid(
        (2,2),
        (0,1)
    )

    ax2.plot(
        df['t'],
        df['v_gps'],
        label='GPS',
        alpha=0.5
    )

    ax2.plot(
        df['t'],
        df['v_obd'],
        label='OBD',
        alpha=0.5
    )

    ax2.plot(
        df['t'],
        states[:,2],
        label='EKF',
        linewidth=2
    )

    ax2.axvspan(
        df.loc[dropout_mask, 't'].min(),
        df.loc[dropout_mask, 't'].max(),
        alpha=0.2,
        color='red',
        label='Dropout'
    )

    ax2.set_title('Speed')

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Speed [m/s]')

    ax2.grid()
    ax2.legend()

    # ORIENTATION

    ax3 = plt.subplot2grid(
        (2,2),
        (1,1)
    )

    yaw_deg = np.degrees(
        np.unwrap(states[:,3])
    )

    yaw_deg = (
        pd.Series(yaw_deg)
        .rolling(
            window=15,
            center=True,
            min_periods=1
        )
        .mean()
    )

    ax3.plot(
        df['t'],
        yaw_deg,
        label='EKF yaw'
    )

    ax3.axvspan(
        df.loc[dropout_mask, 't'].min(),
        df.loc[dropout_mask, 't'].max(),
        alpha=0.2,
        color='red',
        label='Dropout'
    )

    ax3.set_title('Orientation')

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw [deg]')

    ax3.grid()
    ax3.legend()

    plt.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


# MAIN

if __name__ == '__main__':

    GNSS_START = "17:26"
    GNSS_END   = "17:27"

    OBD_START = "17:26"
    OBD_END   = "17:27"

    # LOAD DATA

    df = load_data('data1.csv')

    df = process_time(df)

    df = latlon_to_meters(df)

    df = prepare_speed(df)

    df = prepare_initial_yaw(df)

    # TEST 1:
    # GNSS DROPOUT

    df_gnss, gnss_mask = apply_dropout(
        df,
        mode="GNSS",
        start_time=GNSS_START,
        end_time=GNSS_END
    )

    states_gnss = run_filter(df_gnss)

    plot_test(
        df_gnss,
        states_gnss,
        gnss_mask,
        'GNSS Dropout Test'
    )

    # TEST 2:
    # OBD DROPOUT

    df_obd, obd_mask = apply_dropout(
        df,
        mode="OBD",
        start_time=OBD_START,
        end_time=OBD_END
    )

    states_obd = run_filter(df_obd)

    plot_test(
        df_obd,
        states_obd,
        obd_mask,
        'OBD Dropout Test'
    )
