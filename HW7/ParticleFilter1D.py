import numpy as np
import matplotlib.pyplot as plt

#Параметры
N          = 100    # частиц
STEPS      = 40     # шагов
PROC_STD   = 0.3    # шум движения (непредсказуемость объекта)
OBS_STD    = 0.8    # шум датчика  (Недостоверность измерений)
VELOCITY   = 0.5    # скорость объекта

rng = np.random.default_rng(42) #rng seed. Это чисто для воспроизводимости. Для полностью рандомного варианта надо убрать

#Инициализация
true_pos  = 0.0
particles = rng.uniform(-2, 2, N)   # Наши изначальные гипотезы. диапозон -2,2. Равномерное распределение

history = {"true": [], "obs": [], "estimate": [], "particles": []}

#Главный цикл
for t in range(STEPS):
    #PREDICT (двигаем каждую частицу и добавляем шум)
    particles = particles + VELOCITY + rng.normal(0, PROC_STD, N)
    #TRUE STATE + OBSERVATION
    true_pos += VELOCITY
    obs = true_pos + rng.normal(0, OBS_STD) # шумный датчик дает шумные данные
    #UPDATE (считаем вес для каждой частицы)
    weights = np.exp(-0.5 * ((particles - obs) / OBS_STD) ** 2)
    weights /= weights.sum()   # нормировка
    #ESTIMATE (считаем взвешенное среднее)
    estimate = np.sum(weights * particles)
    #RESAMPLE (пересоздаем облако по весам)
    idx       = rng.choice(N, size=N, p=weights) #Выбираем из исходных частиц новый набор (Самые вероятные точки будут выбраны много раз, а самые маловероятные не будут выбраны вообще)
    particles = particles[idx]
    # сохраняем
    history["true"].append(true_pos)
    history["obs"].append(obs)
    history["estimate"].append(estimate)
    history["particles"].append(particles.copy())

#Визуализация
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("Particle Filter 1D", color="white", fontsize=13)
for ax in (ax1, ax2):
    ax.tick_params(colors="#aaa")
    ax.grid(color="#1e2130", linewidth=0.7)
    for s in ax.spines.values():
        s.set_edgecolor("#1e2130")

steps = range(STEPS)

#траектории

#частицы (scatter по всем шагам)
for t in steps:
    ax1.scatter([t] * N, history["particles"][t],
                s=1.5, alpha=0.08, color="#EF9F27")

ax1.plot(steps, history["true"],     color="#E05A3A", lw=2,   label="Истинная позиция")
ax1.plot(steps, history["obs"],      color="#7F77DD", lw=1,   ls="--", alpha=0.7, label="Наблюдение (датчик)")
ax1.plot(steps, history["estimate"], color="#1D9E75", lw=2,   label="Оценка фильтра")
ax1.set_ylabel("позиция", color="#aaa")
ax1.legend(fontsize=9)

#ошибки
error_obs = np.abs(np.array(history["obs"])      - np.array(history["true"]))
error_pf  = np.abs(np.array(history["estimate"]) - np.array(history["true"]))

ax2.plot(steps, error_obs, color="#7F77DD", lw=1.5, ls="--", label=f"Ошибка датчика  (среднее {error_obs.mean():.2f})")
ax2.plot(steps, error_pf,  color="#1D9E75", lw=2,   label=f"Ошибка фильтра (среднее {error_pf.mean():.2f})")
ax2.set_ylabel("ошибка", color="#aaa")
ax2.set_xlabel("шаг", color="#aaa")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("pf_1D.png", dpi=130, bbox_inches="tight")
plt.show()