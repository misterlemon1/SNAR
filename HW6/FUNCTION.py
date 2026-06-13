import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt
from QUAT import IMU, Quaternion

def load_csv(filepath):
    meta  = pd.read_csv(filepath, nrows=0).columns.tolist()
    label = meta[-1] if meta else ""
    df    = pd.read_csv(filepath, skiprows=1)
    df.columns = df.columns.str.strip()

    if "imuTimestamp" in df.columns:
        t = df["imuTimestamp"].astype(float)
        df["time_s"] = (t - t.iloc[0]) / 1e3          # мс → с
    elif "utcTimestamp(millis)" in df.columns:
        t = df["utcTimestamp(millis)"].astype(float)
        df["time_s"] = (t - t.iloc[0]) / 1e3           # мс → с
    else:
        df["time_s"] = np.arange(len(df)) * 0.01       # fallback

    return df, label

def fmt(ax, title, ylabel):
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Время, с", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

def Processing(maindir,SESSION_LABEL="",SAVE_PATH=""):
    G = 9.81
    ACC_FILE = maindir + "/accelerometer_1.csv"
    GYRO_FILE = maindir + "/gyroscope_2.csv"
    PRESSURE_FILE = maindir + "/pressure_8.csv"
    acc_df,  lbl_acc  = load_csv(ACC_FILE)
    gyro_df, lbl_gyro = load_csv(GYRO_FILE)
    pres_df, lbl_pres = load_csv(PRESSURE_FILE)
    raw_t = gyro_df["imuTimestamp"].astype(float).values
    wx_v  = gyro_df["ang_vel_x[rad/s]"].values
    wy_v  = gyro_df["ang_vel_y[rad/s]"].values
    wz_v  = gyro_df["ang_vel_z[rad/s]"].values
    t_gyr = gyro_df["time_s"].values
    imu_obj = IMU()
    t_out, qi_arr, qj_arr, qk_arr = [], [], [], []
    for i in range(1, len(raw_t)):
        dt = (raw_t[i] - raw_t[i - 1]) / 1e3   # мс → с
        if dt <= 0 or dt > 1.0:
            continue
        imu_obj.update(wx_v[i], wy_v[i], wz_v[i], dt)
        q = imu_obj.direction
        t_out.append(t_gyr[i])
        qi_arr.append(q.i)
        qj_arr.append(q.j)
        qk_arr.append(q.k)
    t_out  = np.array(t_out)
    qi_arr = np.array(qi_arr)
    qj_arr = np.array(qj_arr)
    qk_arr = np.array(qk_arr)
    t_acc_raw = acc_df["time_s"].values
    ax_v = acc_df["acc_x[m/s^2]"].astype(float).values
    ay_v = acc_df["acc_y[m/s^2]"].astype(float).values
    az_v = acc_df["acc_z[m/s^2]"].astype(float).values
    ax_i = np.interp(t_gyr, t_acc_raw, ax_v)
    ay_i = np.interp(t_gyr, t_acc_raw, ay_v)
    az_i = np.interp(t_gyr, t_acc_raw, az_v)
    w_norm    = np.sqrt(wx_v**2 + wy_v**2 + wz_v**2)
    still_thr = np.percentile(w_norm, 1)
    still     = w_norm < still_thr
    g_vec_raw = np.array([ax_i[still].mean(), ay_i[still].mean(), az_i[still].mean()])
    g_hat     = g_vec_raw / np.linalg.norm(g_vec_raw)
    acc_bias = np.array([
        ax_i[still].mean() - g_hat[0] * G,
        ay_i[still].mean() - g_hat[1] * G,
        az_i[still].mean() - g_hat[2] * G,
    ])
    # Проход 1: вращаем a_body в мировую СК
    aw_x = np.zeros(len(t_gyr))
    aw_y = np.zeros(len(t_gyr))
    aw_z = np.zeros(len(t_gyr))
    imu2 = IMU()
    for i in range(1, len(raw_t)):
        dt = (raw_t[i] - raw_t[i - 1]) / 1e3
        if dt <= 0 or dt > 1.0:
            continue
        imu2.update(wx_v[i], wy_v[i], wz_v[i], dt)
        q     = imu2.direction
        q_inv = q.inverse()
        a_body = np.array([ax_i[i] - acc_bias[0],
                           ay_i[i] - acc_bias[1],
                           az_i[i] - acc_bias[2]])
        a_pure  = Quaternion([0.0, *a_body])
        a_rot   = q * a_pure * q_inv
        aw_x[i] = a_rot.i
        aw_y[i] = a_rot.j
        aw_z[i] = a_rot.k

    # Вектор гравитации в мировой СК (из покойных моментов)
    g_world_vec = np.array([aw_x[still].mean(), aw_y[still].mean(), aw_z[still].mean()])
    # Проход 2: вычитаем g_world_vec → a_dyn
    ad_x = np.zeros(len(t_gyr))
    ad_y = np.zeros(len(t_gyr))
    ad_z = np.zeros(len(t_gyr))
    imu2 = IMU()
    for i in range(1, len(raw_t)):
        dt = (raw_t[i] - raw_t[i - 1]) / 1e3
        if dt <= 0 or dt > 1.0:
            continue
        imu2.update(wx_v[i], wy_v[i], wz_v[i], dt)
        q     = imu2.direction
        q_inv = q.inverse()
        a_body = np.array([ax_i[i] - acc_bias[0],
                           ay_i[i] - acc_bias[1],
                           az_i[i] - acc_bias[2]])
        a_pure  = Quaternion([0.0, *a_body])
        a_rot   = q * a_pure * q_inv
        a_world = np.array([a_rot.i, a_rot.j, a_rot.k])
        a_dyn   = a_world - g_world_vec
        ad_x[i] = a_dyn[0]
        ad_y[i] = a_dyn[1]
        ad_z[i] = a_dyn[2]
    P   = pres_df["pressure[millibar]"].astype(float).values
    t_p = pres_df["time_s"].values
    P0      = P[0]
    alt_raw = 44330.0 * (1.0 - (P / P0) ** (1.0 / 5.255))   # формула из Carusel.py
    win_alt = min(11, len(alt_raw) if len(alt_raw) % 2 == 1 else len(alt_raw) - 1)
    alt_flt = medfilt(alt_raw, win_alt)
    CLR = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db", "mag": "#2c3e50"}

    pos_x = np.zeros(len(t_gyr))
    pos_y = np.zeros(len(t_gyr))
    pos_z = np.zeros(len(t_gyr))
    ZERO_VEL_THR = 0.05
    VEL_DECAY    = 0.995
    vel        = np.zeros(3)
    a_dyn_prev = np.zeros(3)
    vel_prev   = np.zeros(3)

    for i in range(1, len(raw_t)):
        dt = (raw_t[i] - raw_t[i - 1]) / 1e3
        if dt <= 0 or dt > 1.0:
            continue
        a_dyn_cur = np.array([ad_x[i], ad_y[i], ad_z[i]])
        if w_norm[i] < ZERO_VEL_THR:
            vel[:] = 0.0
        else:
            vel *= VEL_DECAY
        vel_prev = vel.copy()
        vel += 0.5 * (a_dyn_prev + a_dyn_cur) * dt
        a_dyn_prev = a_dyn_cur.copy()
        pos_x[i] = pos_x[i - 1] + 0.5 * (vel_prev[0] + vel[0]) * dt
        pos_y[i] = pos_y[i - 1] + 0.5 * (vel_prev[1] + vel[1]) * dt
        pos_z[i] = pos_z[i - 1] + 0.5 * (vel_prev[2] + vel[2]) * dt

    DISP_WIN = 101
    pos_x_f = medfilt(pos_x, min(DISP_WIN, len(pos_x) if len(pos_x) % 2 == 1 else len(pos_x) - 1))
    pos_y_f = medfilt(pos_y, min(DISP_WIN, len(pos_y) if len(pos_y) % 2 == 1 else len(pos_y) - 1))
    pos_z_f = medfilt(pos_z, min(DISP_WIN, len(pos_z) if len(pos_z) % 2 == 1 else len(pos_z) - 1))

    title = SESSION_LABEL or lbl_acc or lbl_gyro or lbl_pres or "IMU Session"
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    gs = gridspec.GridSpec(4, 4, hspace=0.55, wspace=0.35)

    # 1. Линейное ускорение
    ax = fig.add_subplot(gs[0, :2])
    t_acc = acc_df["time_s"]
    ax_x = acc_df["acc_x[m/s^2]"].astype(float)
    ax_y = acc_df["acc_y[m/s^2]"].astype(float)
    ax_z = acc_df["acc_z[m/s^2]"].astype(float)
    ax.plot(t_acc, ax_x, color=CLR["x"], lw=0.9, label="X")
    ax.plot(t_acc, ax_y, color=CLR["y"], lw=0.9, label="Y")
    ax.plot(t_acc, ax_z, color=CLR["z"], lw=0.9, label="Z")
    ax.plot(t_acc, np.sqrt(ax_x ** 2 + ax_y ** 2 + ax_z ** 2),
            color=CLR["mag"], lw=1.1, ls="--", alpha=0.6, label="|a|")
    fmt(ax, "Линейное ускорение (акселерометр)", "м/с²")

    # 2. Угловые скорости
    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(t_gyr, wx_v, color=CLR["x"], lw=0.9, label="ωx")
    ax.plot(t_gyr, wy_v, color=CLR["y"], lw=0.9, label="ωy")
    ax.plot(t_gyr, wz_v, color=CLR["z"], lw=0.9, label="ωz")
    fmt(ax, "Угловые скорости (гироскоп)", "рад/с")

    # 3. Атмосферное давление
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t_p, P, color="#8e44ad", lw=0.9, label="Давление")
    fmt(ax, "Атмосферное давление", "мбар")

    # 4. Барометрическая высота
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(t_p, alt_raw, color="#bdc3c7", lw=0.7, alpha=0.6, label="сырая")
    ax.plot(t_p, alt_flt, color="#e67e22", lw=1.4, label=f"сглаженная (medfilt {win_alt})")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    fmt(ax, "Относительная высота (барометр, нулевой отсчёт = старт)", "Δвысота, м")

    # 5. Мнимая часть кватерниона
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t_out, qi_arr, color=CLR["x"], lw=0.9, label="qi")
    ax.plot(t_out, qj_arr, color=CLR["y"], lw=0.9, label="qj")
    ax.plot(t_out, qk_arr, color=CLR["z"], lw=0.9, label="qk")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylim(-1.1, 1.1)
    fmt(ax, "Мнимая часть кватерниона (qi, qj, qk)", "")

    # 5b. Траектория XY
    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(pos_x_f, pos_y_f, color="#9b59b6", lw=1.4, label="траектория XY")
    ax.scatter([0], [0], color="#2ecc71", zorder=5, s=60, label="старт")
    ax.scatter([pos_x_f[-1]], [pos_y_f[-1]], color="#e74c3c", zorder=5, s=60, label="конец")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("X, м", fontsize=9)
    ax.set_ylabel("Y, м", fontsize=9)
    ax.set_title("Траектория XY (горизонтальная плоскость)", fontsize=11, pad=6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # 6. Ускорение в мировой СК
    ax = fig.add_subplot(gs[3, :2])
    ax.plot(t_gyr, aw_x, color=CLR["x"], lw=0.9, alpha=0.8, label="a_world X")
    ax.plot(t_gyr, aw_y, color=CLR["y"], lw=0.9, alpha=0.8, label="a_world Y")
    ax.plot(t_gyr, aw_z, color=CLR["z"], lw=0.9, alpha=0.8, label="a_world Z")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    fmt(ax, "Ускорение в мировой СК  (a_world = q · a_body · q⁻¹)", "м/с²")

    # 7. Динамическое ускорение
    ax = fig.add_subplot(gs[3, 2:])
    ax.plot(t_gyr, ad_x, color=CLR["x"], lw=0.9, alpha=0.8, label="a_dyn X")
    ax.plot(t_gyr, ad_y, color=CLR["y"], lw=0.9, alpha=0.8, label="a_dyn Y")
    ax.plot(t_gyr, ad_z, color=CLR["z"], lw=0.9, alpha=0.8, label="a_dyn Z")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    fmt(ax, "Динамическое ускорение  (a_world − g_world)", "м/с²")

    plt.tight_layout()
    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
        print(f"\nСохранено: {SAVE_PATH}")
    else:
        plt.show()

