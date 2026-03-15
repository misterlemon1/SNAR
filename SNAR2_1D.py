import numpy as np
import SNAR2_DrawingTools as tools

#Начальные условия

#Параметры мира
N=10 #Длина мира
wdet=[]#Определенность мира (пустой-рандом, иначе будет определено)


#Параметры робота
determ=-1 #Определенность положения (-1 - равновероятное положение, d - везде 0 кроме d%N клетки)\
sensePerMove=2#Число сканов между перемещениями
pHitReal=0.85#Реальная вероятность успешного скана
pHit=0.8 #Сенсор вернет правильное значение (Наше предположение)
pMiss=0.15 #Сенсор вернет неправильное значение (Наше предположение)
pU={
    0:0.7,
    1:0.15,
    -1:0.15
}#Вероятности отклонения при смещении



iterations=5 #Число итераций цикла



def move(p,pU,u):
    "Передвижение на u шагов вправо(отрицательное-влево)"
    pu=np.zeros(N)
    for i in pU.keys():
        pu[(u+i)%N]=pU[i]
    new_p=np.zeros(N)
    for pos in p:
        new_p+=pos*pu
        pu=np.roll(pu,1)
    return new_p

def sense(pos,col,pH,pM,world):
    "Корректирует распределение вероятности, основываясь на цвете и карте"
    new_p=np.zeros(N)
    for i in range(len(world)):
        new_p[i]=pos[i]*(pH*(world[i]==col)+pM*(world[i]!=col))
    new_p /= np.sum(new_p) # нормализация (сумма = 1)
    return new_p


#1d мир
if len(wdet)==0:
    w=np.random.choice(["g","r"],p=[0.5,0.5],size=N)#p=[вероятность зеленого, вероятность красного]
else:
    w=[wdet[i%len(wdet)] for i in range(N)]
#Положение робота
if determ<0:
    position = np.array([1 / N] * N)
else:
    position = np.array([0] * N)
    position[determ % N]=1

history=position # для построения тепловой карты вероятности нахождения робота
rp=[0]*N
if determ<0:
    idx = np.random.randint(0,N)
    rp[idx]=1
else:
    rp[determ%N]=1
    idx=determ%N

history_real=np.array(rp)

#Главный цикл
for _ in range(iterations):
    u=np.random.randint(-N,N+1)
    du=np.random.choice(list(pU.keys()), p=list(pU.values()))
    idx = (idx + u + du) % N
    rp=np.roll(rp,u+du)
    position=move(position, pU, u)
    history=np.vstack((history, position))
    history_real=np.vstack((history_real,rp))
    pCol=[float(pHitReal * (w[idx] == "r") + (1 - pHitReal) * (w[idx] != "r")),
     float(pHitReal * (w[idx] == "g") + (1 - pHitReal) * (w[idx] != "g"))]
    for _ in range(sensePerMove):
        col = np.random.choice(["r", "g"], p=pCol)#Определяем цвет с датчика
        position = sense(position, col, pHit, pMiss, w)
    history = np.vstack((history, position))
    history_real = np.vstack((history_real, rp))
tools.wMap1D(w,56,2,"1DWorld")
tools.hMap(history,56,2,"hMap",True,history_real)