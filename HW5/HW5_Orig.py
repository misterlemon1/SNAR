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


# =========================================================
# LAT/LON -> METERS
# =========================================================

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

    x = (
        (lon - lon0)
        * R
        * np.cos(lat0)
    )

    y = (
        (lat - lat0)
        * R
    )

    df['x'] = x
    df['y'] = y

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


def simulate_dropout_by_time(df):

    df = df.copy()

    df[' Device Time'] = pd.to_datetime(
        df[' Device Time']
    )


    gps_mask = (
        (
            df[' Device Time'].dt.time >=
            pd.to_datetime("17:25").time()
        )
        &
        (
            df[' Device Time'].dt.time <=
            pd.to_datetime("17:26").time()
        )
    )

    df.loc[gps_mask, 'x'] = np.nan
    df.loc[gps_mask, 'y'] = np.nan


    obd_mask = (
        (
            df[' Device Time'].dt.time >=
            pd.to_datetime("17:40").time()
        )
        &
        (
            df[' Device Time'].dt.time <=
            pd.to_datetime("17:41").time()
        )
    )

    df.loc[obd_mask, 'v_obd'] = np.nan

    return df


def normalize_angle(angle):

    return (
        (angle + np.pi)
        % (2*np.pi)
        - np.pi
    )


class EKF:

    def __init__(self, dt):

        self.dt = dt

        # state:
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

        # limit abrupt jumps

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


        if not np.isnan(df['v_obd'].iloc[i]):

            ekf.update_obd(
                df['v_obd'].iloc[i]
            )


        states.append(
            ekf.x.flatten()
        )

    return np.array(states)


def plot_results(df, states):

    plt.figure(figsize=(16,7))


    ax1 = plt.subplot2grid(
        (2,2),
        (0,0),
        rowspan=2
    )

    ax1.plot(
        df['x'],
        df['y'],
        label='GPS',
        alpha=0.5
    )

    ax1.plot(
        states[:,0],
        states[:,1],
        label='EKF',
        linewidth=2
    )

    ax1.set_title('Trajectory')

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    ax1.axis('equal')

    ax1.grid()
    ax1.legend()

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

    ax2.set_title('Speed')

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Speed [m/s]')

    ax2.grid()
    ax2.legend()

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

    ax3.set_title('Orientation')

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw [deg]')

    ax3.grid()
    ax3.legend()

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    df = load_data('data1.csv')

    df = process_time(df)

    df = latlon_to_meters(df)

    df = prepare_speed(df)

    df = prepare_initial_yaw(df)

    df = simulate_dropout_by_time(df)

    states = run_filter(df)

    plot_results(df, states)