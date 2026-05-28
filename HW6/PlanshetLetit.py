import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from QUAT import IMU, Quaternion

#Загрузка

def load_csv(filename):
    df = pd.read_csv(filename, skiprows=1)
    df.columns = df.columns.str.strip()
    print(f"  {filename}: {len(df)} строк")
    return df

print("=== Загрузка данных ===")
acc = load_csv("Rotate_Fly/2026-05-28_22-10-22.697_Active 10 Pro/accelerometer_1.csv")
gyr = load_csv("Rotate_Fly/2026-05-28_22-10-22.697_Active 10 Pro/gyroscope_2.csv")
prs = load_csv("Rotate_Fly/2026-05-28_22-10-22.697_Active 10 Pro/pressure_8.csv")

#Нормализация времени

t0    = min(acc['imuTimestamp'].iloc[0], gyr['imuTimestamp'].iloc[0], prs['imuTimestamp'].iloc[0])
t_acc = (acc['imuTimestamp'].values - t0) / 1e3
t_gyr = (gyr['imuTimestamp'].values - t0) / 1e3
t_prs = (prs['imuTimestamp'].values - t0) / 1e3

ax_v = acc['acc_x[m/s^2]'].values
ay_v = acc['acc_y[m/s^2]'].values
az_v = acc['acc_z[m/s^2]'].values

wx_v = gyr['ang_vel_x[rad/s]'].values
wy_v = gyr['ang_vel_y[rad/s]'].values
wz_v = gyr['ang_vel_z[rad/s]'].values

P = prs['pressure[millibar]'].values

#Проверка временных меток

def check_timestamps(t, name):
    dt = np.diff(t)
    bad = (dt <= 0).sum() + (dt > 1.0).sum()
    print(f"  {name}: dt mean={dt.mean():.4f}s  {'✓ OK' if not bad else f'⚠ аномалий: {bad}'}")

print("\n=== Проверка временных меток ===")
check_timestamps(t_acc, "Акселерометр")
check_timestamps(t_gyr, "Гироскоп")
check_timestamps(t_prs, "Барометр")

'''
Интерполяция акселерометра на сетку гироскопа
Гироскоп и акселерометр могут иметь слегка разные временные метки,
интерполируем акселерометр на моменты гироскопа для синхронной обработки.
'''

ax_interp = np.interp(t_gyr, t_acc, ax_v)
ay_interp = np.interp(t_gyr, t_acc, ay_v)
az_interp = np.interp(t_gyr, t_acc, az_v)

# Strapdown-алгоритм

'''
На каждом шаге:
1. Обновляем кватернион ориентации по гироскопу (как раньше)
2. Вектор гравитации в глобальной СК: g_global = [0, 0, 9.81]
3. Поворачиваем его в систему телефона: g_body = q^-1 * g_global * q
4. Вычитаем из показаний акселерометра: a_lin_body = a_meas - g_body
5. Поворачиваем линейное ускорение в глобальную СК: a_lin_global = q * a_lin_body * q^-1
6. Интегрируем скорость и перемещение в глобальной СК
Дополнительно: обнуление скорости (ZUPT) в моменты покоя —
детектируем покой по малой норме акселерометра относительно g
и малой угловой скорости.
'''

print("\n=== Strapdown-интегрирование ===")

G = 9.81

raw_t = gyr['imuTimestamp'].values

# Выходные массивы
t_out  = []
rolls, pitches, yaws     = [], [], []
qi_arr, qj_arr, qk_arr   = [], [], []
vx_arr, vy_arr, vz_arr   = [], [], []
px_arr, py_arr, pz_arr   = [], [], []
a_gx_arr, a_gy_arr, a_gz_arr = [], [], []  # линейное ускорение в глобальной СК

imu_obj = IMU()

# Начальные скорость и позиция
vx, vy, vz = 0.0, 0.0, 0.0
px, py, pz = 0.0, 0.0, 0.0

# Порог детектора покоя
ACC_STILL_THR = 0.3   #отклонение |a| от g
GYR_STILL_THR = 0.05  #норма угловой скорости

for i in range(1, len(raw_t)):
    dt = (raw_t[i] - raw_t[i-1]) / 1e3
    if dt <= 0 or dt > 1.0:
        continue

    # 1. Обновляем ориентацию
    imu_obj.update(wx_v[i], wy_v[i], wz_v[i], dt)
    q = imu_obj.direction          # текущий кватернион (единичный)

    # 2. Вектор гравитации в глобальной СК как чистый кватернион
    g_global = Quaternion([0, 0, 0, G])

    # 3. Гравитация в системе тела: g_body = q^-1 * g_global * q
    g_body = q.inverse() * g_global * q

    # 4. Линейное ускорение в системе тела
    a_bx = ax_interp[i] - g_body.i
    a_by = ay_interp[i] - g_body.j
    a_bz = az_interp[i] - g_body.k

    # 5. Линейное ускорение в глобальной СК
    a_body_q  = Quaternion([0, a_bx, a_by, a_bz])
    a_glob_q  = q * a_body_q * q.inverse()
    agx, agy, agz = a_glob_q.i, a_glob_q.j, a_glob_q.k

    # 6. ZUPT — детектор покоя
    acc_norm = np.sqrt(ax_interp[i]**2 + ay_interp[i]**2 + az_interp[i]**2)
    gyr_norm = np.sqrt(wx_v[i]**2 + wy_v[i]**2 + wz_v[i]**2)
    is_still = (abs(acc_norm - G) < ACC_STILL_THR) and (gyr_norm < GYR_STILL_THR)

    if is_still:
        vx, vy, vz = 0.0, 0.0, 0.0
    else:
        # Интеграция скорости (метод Эйлера)
        vx += agx * dt
        vy += agy * dt
        vz += agz * dt

    # Интеграция перемещения
    px += vx * dt
    py += vy * dt
    pz += vz * dt

    t_out.append(t_gyr[i])
    r, p, y = imu_obj.to_euler_deg()
    rolls.append(r); pitches.append(p); yaws.append(y)
    qi_arr.append(q.i); qj_arr.append(q.j); qk_arr.append(q.k)
    vx_arr.append(vx); vy_arr.append(vy); vz_arr.append(vz)
    px_arr.append(px); py_arr.append(py); pz_arr.append(pz)
    a_gx_arr.append(agx); a_gy_arr.append(agy); a_gz_arr.append(agz)

t_out  = np.array(t_out)
rolls  = np.array(rolls);  pitches = np.array(pitches); yaws = np.array(yaws)
qi_arr = np.array(qi_arr); qj_arr  = np.array(qj_arr);  qk_arr = np.array(qk_arr)
vx_arr = np.array(vx_arr); vy_arr  = np.array(vy_arr);  vz_arr = np.array(vz_arr)
px_arr = np.array(px_arr); py_arr  = np.array(py_arr);  pz_arr = np.array(pz_arr)
a_gx_arr = np.array(a_gx_arr)
a_gy_arr = np.array(a_gy_arr)
a_gz_arr = np.array(a_gz_arr)

altitude = 44330.0 * (1.0 - (P / P[0]) ** (1.0 / 5.255))

print(f"  Шагов: {len(t_out)},  длительность: {t_out[-1]:.1f} с")
print(f"  Норма кватерниона в конце: {imu_obj.direction.norm():.6f}")
print(f"  Итоговое перемещение: x={px_arr[-1]:.2f} м  y={py_arr[-1]:.2f} м  z={pz_arr[-1]:.2f} м")


fig, axes = plt.subplots(5, 2, figsize=(18, 20))
fig.suptitle("IMU Data Analysis — strapdown с компенсацией g через кватернион", fontsize=13)

def finish(ax, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Время, с", fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)


ax = axes[0][0]
ax.plot(t_acc, ax_v, label='ax')
ax.plot(t_acc, ay_v, label='ay')
ax.plot(t_acc, az_v, label='az')
ax.axhline(9.81, color='gray', linestyle=':', linewidth=0.8, label='g=9.81')
finish(ax, "Акселерометр", "m/s²")

ax = axes[1][0]
ax.plot(t_gyr, wx_v, label='ωx')
ax.plot(t_gyr, wy_v, label='ωy')
ax.plot(t_gyr, wz_v, label='ωz')
finish(ax, "Гироскоп", "рад/с")

ax = axes[2][0]
ax.plot(t_out, rolls,   label='Roll (крен)')
ax.plot(t_out, pitches, label='Pitch (тангаж)')
ax.plot(t_out, yaws,    label='Yaw (рыскание)')
finish(ax, "Ориентация — углы Эйлера", "градусы")

ax = axes[3][0]
ax.plot(t_prs, P, label='давление')
finish(ax, "Давление", "мбар")

ax = axes[4][0]
ax.plot(t_prs, altitude, label='высота')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
finish(ax, "Относительная высота из давления", "метры")

ax = axes[0][1]
ax.plot(t_out, a_gx_arr, label='a_x (глоб.)')
ax.plot(t_out, a_gy_arr, label='a_y (глоб.)')
ax.plot(t_out, a_gz_arr, label='a_z (глоб.)')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
finish(ax, "Линейное ускорение в глобальной СК (без g)", "m/s²")

ax = axes[1][1]
ax.plot(t_out, vx_arr, label='vx')
ax.plot(t_out, vy_arr, label='vy')
ax.plot(t_out, vz_arr, label='vz')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
finish(ax, "Скорость в глобальной СК (ZUPT)", "м/с")

ax = axes[2][1]
ax.plot(t_out, qi_arr, label='qi')
ax.plot(t_out, qj_arr, label='qj')
ax.plot(t_out, qk_arr, label='qk')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
finish(ax, "Мнимая часть кватерниона (qi, qj, qk)", "")

ax = axes[3][1]
ax.plot(t_out, px_arr, label='x')
ax.plot(t_out, py_arr, label='y')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
finish(ax, "Относительное перемещение (горизонталь)", "метры")

ax = axes[4][1]
ax.plot(t_out, pz_arr,   label='z (акс. strapdown)', color='tab:green')
ax.plot(t_prs, altitude, label='z (барометр)',        color='tab:orange', linewidth=1.2)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
finish(ax, "Относительное перемещение (вертикаль)", "метры")

plt.tight_layout()
plt.savefig("Rotate_Fly/imu_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ График сохранён: imu_analysis.png")