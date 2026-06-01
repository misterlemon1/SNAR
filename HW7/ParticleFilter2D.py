import numpy as np
import matplotlib.pyplot as plt

#Параметры
MODE     = "curved"  # "linear" или "curved" Влияет только на режим работы
N        = 200
STEPS    = 80
PROC_STD = 0.15  # шум движения
OBS_STD  = 0.5   # шум датчика

rng = np.random.default_rng(42) #Сид рнг. Всё так же нужен только для повторяемости

#Инициализация
true_pos = np.array([0.0, 0.0])
velocity = np.array([0.3, 0.1]) # начальная скорость (используется в обоих режимах)
particles = rng.uniform(-2, 2, size=(N, 2))
history = {"true": [], "obs": [], "estimate": [], "particles": []}

#Главный цикл
for t in range(STEPS):
    #Движение
    if MODE == "linear":
        true_pos = true_pos + velocity
    elif MODE == "curved":
        #Поворот скорости на некий угол каждый шаг
        angle = 0.08   # радиан за шаг (Можно попробовать с этим поделать что-нибудь)
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])  # матрица поворота
        velocity = rot @ velocity   # поворачиваем вектор скорости (@-матричное умножение)
        true_pos = true_pos + velocity
    obs = true_pos + rng.normal(0, OBS_STD, size=2)  # шумный датчик
    #PREDICT
    #Важно: частицы не знают про поворот — они просто добавляют шум.
    #А это значит что шум процесса (PROC_STD) должен быть достаточно большим,
    #чтобы покрыть неожиданные манёвры объекта
    particles = particles + velocity + rng.normal(0, PROC_STD, size=(N, 2))
    #UPDATE
    dist = np.linalg.norm(particles - obs, axis=1)
    weights = np.exp(-0.5 * (dist / OBS_STD) ** 2)
    weights /= weights.sum()
    #ESTIMATE
    estimate = np.sum(weights[:, None] * particles, axis=0)
    #RESAMPLE
    idx = rng.choice(N, size=N, p=weights)
    particles = particles[idx]

    history["true"].append(true_pos.copy())
    history["obs"].append(obs.copy())
    history["estimate"].append(estimate.copy())
    history["particles"].append(particles.copy())


true_arr = np.array(history["true"])
obs_arr  = np.array(history["obs"])
est_arr  = np.array(history["estimate"])

#Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(f"Particle Filter 2D — режим: {MODE}", fontsize=13)

for ax in (ax1, ax2):
    ax.tick_params(colors="#aaa")
    ax.grid(color="#1e2130", linewidth=0.7)
    for s in ax.spines.values():
        s.set_edgecolor("#1e2130")

# частицы — снимки каждые 15 шагов
for t in range(0, STEPS, 15):
    pts = history["particles"][t]
    ax1.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.15, color="#EF9F27")

ax1.plot(obs_arr[:, 0],  obs_arr[:, 1],  color="#7F77DD", lw=1, ls="--", alpha=0.7, label="Наблюдения (датчик)")
ax1.plot(true_arr[:, 0], true_arr[:, 1], color="#E05A3A", lw=2, label="Истинный путь")
ax1.plot(est_arr[:, 0],  est_arr[:, 1],  color="#1D9E75", lw=2, label="Оценка фильтра")
ax1.plot(*true_arr[0],  "o", color="#E05A3A", ms=9)
ax1.plot(*true_arr[-1], "s", color="#E05A3A", ms=9)
ax1.text(true_arr[0, 0] + 0.1, true_arr[0, 1], "старт", color="#E05A3A", fontsize=8)
ax1.set_aspect("equal")
ax1.set_xlabel("x", color="#aaa")
ax1.set_ylabel("y", color="#aaa")
ax1.legend(fontsize=9)

error_obs = np.linalg.norm(obs_arr - true_arr, axis=1)
error_pf  = np.linalg.norm(est_arr - true_arr, axis=1)
steps = range(STEPS)
ax2.plot(steps, error_obs, color="#7F77DD", lw=1.5, ls="--", label=f"Ошибка датчика  (среднее {error_obs.mean():.2f})")
ax2.plot(steps, error_pf,  color="#1D9E75", lw=2, label=f"Ошибка фильтра (среднее {error_pf.mean():.2f})")
ax2.set_xlabel("шаг", color="#aaa")
ax2.set_ylabel("ошибка", color="#aaa")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"pf_2d_{MODE}.png", dpi=130, bbox_inches="tight")
plt.show()