import matplotlib.pyplot as plt
import folium
from HW3_BaseFunc import load_data, process_time, latlon_to_meters

df = load_data()
process_time(df)
latlon_to_meters(df) #Подготовка всего необходимого

### Траектория на графике

plt.figure(figsize=(8, 8))
plt.plot(df['x'], df['y'], linewidth=2)

plt.scatter(df['x'].iloc[0], df['y'].iloc[0], label='Start')
plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], label='End')

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("Trajectory (meters)")
plt.legend()
plt.grid()

plt.axis('equal')  # чтоб красиво было
plt.show()

### Траектория на карте

start_lat = df[' Latitude'].iloc[0]
start_lon = df[' Longitude'].iloc[0]

m = folium.Map(location=[start_lat, start_lon], zoom_start=15)
coords = list(zip(df[' Latitude'], df[' Longitude'])) # точки маршрута
folium.PolyLine(coords, weight=3).add_to(m) # линия траектории
folium.Marker(coords[0], tooltip="Start").add_to(m) # старт
folium.Marker(coords[-1], tooltip="End").add_to(m) # конец

#Для сохранения карты нужна команда ниже
m.save("trajectory.html")
