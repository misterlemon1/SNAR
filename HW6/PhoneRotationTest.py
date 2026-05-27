from QUAT import *
import pandas as pd
import matplotlib.pyplot as plt

df1=pd.read_csv("Testing/Exp1.csv")
df2=pd.read_csv("Testing/Exp2.csv")

IMU1=IMU()
IMU2=IMU()

def build_graph(IMUObj,df):
    tlast=0
    qx=[]
    qy=[]
    qz=[]
    time=[]
    for index,row in df.iterrows():
        wx=row["Gyroscope x (rad/s)"]
        wy=row["Gyroscope y (rad/s)"]
        wz=row["Gyroscope z (rad/s)"]
        t=row["Time (s)"]
        dt=t-tlast
        tlast=t
        time.append(t)
        IMUObj.update(wx,wy,wz,dt)
        a,b,c=IMUObj.to_euler_deg()
        qx.append(a)
        qy.append(b)
        qz.append(c)
    plt.plot(time,qx,"r",label="Поворот вдоль X")
    plt.plot(time,qy,"g",label="Поворот вдоль Y")
    plt.plot(time,qz,"b",label="Поворот вдоль Z")
    plt.xlabel("Время (с)")
    plt.ylabel("Угол (°)")
    plt.legend()
    plt.grid(1)

def build_graphq(IMUObj,df):
    tlast=0
    qx=[]
    qy=[]
    qz=[]
    time=[]
    for index,row in df.iterrows():
        wx=row["Gyroscope x (rad/s)"]
        wy=row["Gyroscope y (rad/s)"]
        wz=row["Gyroscope z (rad/s)"]
        t=row["Time (s)"]
        dt=t-tlast
        tlast=t
        time.append(t)
        IMUObj.update(wx,wy,wz,dt)
        a,b,c=IMUObj.direction.vector3
        qx.append(a)
        qy.append(b)
        qz.append(c)
    plt.plot(time,qx,"r",label="i")
    plt.plot(time,qy,"g",label="j")
    plt.plot(time,qz,"b",label="k")
    plt.xlabel("Время (с)")
    plt.ylabel("значение компоненты")
    plt.legend()
    plt.grid(1)

plt.subplot(2,2,1)
build_graph(IMU1,df1)
plt.subplot(2,2,2)
build_graph(IMU2,df2)
plt.subplot(2,2,3)
build_graphq(IMU1,df1)
plt.subplot(2,2,4)
build_graphq(IMU2,df2)

plt.show()