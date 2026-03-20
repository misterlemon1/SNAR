import numpy as np
import SNAR2_DrawingTools as tools

#Начальные условия

#Параметры мира
Nx=10 #Длина мира
Ny=10 #Ширина мира
wdet=[]#Определитель мира(Повторяющийся паттерн из последовательностей "g","r" или если пусто, то рандом)

#Позиция и перемещение
idx=-1 #Позиция по оси х (Отрицательное значит неопределенность)
idy=-1 #Позиция по оси y (Отрицательное значит неопределенность)



pHitReal=0.85#Реальная вероятность успешного скана
pHit=0.8 #Сенсор вернет правильное значение (Наше предположение)
pMiss=0.15 #Сенсор вернет неправильное значение (Наше предположение)
pUx={
    0:0.8,
    1:0.1,
    -1:0.1
}#Вероятности отклонения при смещении по оси x
pUy={
    0:0.8,
    1:0.1,
    -1:0.1
}#Вероятности отклонения при смещении по оси y

#Цикл локализации
iterations=20 #Число итераций цикла
sensePerMove=1 #Сканов за одно перемещение

#Параметры визуализации
ps=56 #Размер квадрата
lw=2  #Ширина граничной линии
save_period = 10 #период сохранения карт

save_period = save_period if (save_period>0 and save_period<=iterations) else iterations

#Т.к. движение коммутативно порядок не важен(сначала x, потом y или наоборот дадут один результат), будем двигаться сначала по x, затем по y
def movex(p,u,pU):
    pu=np.zeros(Nx)
    for du in pU.keys():
        pu[(u+du)%Nx]+=pU[du]
    new_p=np.zeros(p.shape)
    for i in range(p.size):
        new_p[i//Nx,:]+=p[i//Nx,i%Nx]*pu
        pu=np.roll(pu,1)
    return new_p
def movey(p,u,pU):
    pu=np.zeros(Ny)
    for du in pU.keys():
        pu[(u+du)%Ny]+=pU[du]
    new_p=np.zeros(p.shape)
    for i in range(p.size):
        new_p[:,i//Ny]+=p[i%Ny,i//Ny]*pu
        pu=np.roll(pu,1)
    return new_p

def worldGen(wdet):
    wdet=np.array(wdet)
    if wdet.size==0:
        return(np.random.choice(["r","g","b","p"],p=[0.25,0.25,0.25,0.25],size=(Ny,Nx)))
    w=[]
    for i in range(Ny):
        dw=[]
        for j in range(Nx):
            dw+=[wdet[i%wdet.shape[0]][j%wdet.shape[1]]]
        w+=[dw]
    return np.array(w)

def sense(world, idx, idy):
    colors=np.array(["g","r","b","p"])
    colProb=np.where(colors==world[idy,idx],pHitReal,(1-pHitReal)/3) #Каждый неправильный цвет имеет 1/3 вероятности ошибки скана
    return np.random.choice(colors,p=colProb)

def enchSense(world,idx,idy,radius): #Более крутой скан в квадратной области от него(отход на radius от точки) + возвращает вероятностное поле
    newprob=np.zeros((Ny,Nx))
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            col=sense(world,(idx+dx)%Nx,(idy+dy)%Ny)
            newprob+=np.roll(np.where(world==col,pHit,pMiss),(-dy,-dx),(0,1))
    return newprob

p=np.zeros((Ny,Nx))

#Реальное положение робота
idrx=idx if idx>=0 else np.random.randint(0,Nx)
idry=idy if idy>=0 else np.random.randint(0,Ny)


#Начальные вероятности
if (idx>=0 and idy>=0):
    p[idy,idx]=1
elif (idx>=0):
    p[:,idx]=1/(p[:,0].size)
elif (idy>=0):
    p[idy,:]=1/(p[0,:].size)
else:
    p[:,:]=1/(p[:,:].size)

realPos=np.zeros((Ny,Nx))
realPos[idry,idrx]=1


#Визуализация начальных условий
world=worldGen(wdet)
tools.wMap2D(world,ps,lw,"Map2D_C&S")#Визуализация карты
tools.hMap(p,ps,lw,"probDistInit2D_C&S",True,realPos)#Визуализация распределения вероятностей и реального положения

for itr in range(iterations):
    #Перемещение
    ux=np.random.randint(-Nx,Nx+1)#Перемещение, которое ожидается
    uy=np.random.randint(-Ny,Ny+1)#Перемещение, которое ожидается
    dux=np.random.choice(list(pUx.keys()),p=list(pUx.values()))#Отклонение от запланированного
    duy=np.random.choice(list(pUy.keys()),p=list(pUy.values()))#Отклонение от запланированного
    p = movex(p,ux,pUx)
    realPos[idry,:]=np.roll(realPos[idry,:],ux+dux)
    idrx=(idrx+ux+dux)%Nx #Т.к. пространство зациклено
    p = movey(p,uy,pUy)
    realPos[:, idrx] = np.roll(realPos[:, idrx], uy + duy)
    idry=(idry+uy+duy)%Ny #Т.к. пространство замкнуто
    #Скан
    for _ in range(sensePerMove):
        p*=enchSense(world,idrx,idry,1)
        p/=np.sum(p)
    if itr%save_period==0:#Сохранение кратных итераций
        tools.hMap(p, ps, lw, f"probDistIter_{itr}_2D_C&S", True, realPos, True)
#Итог
tools.hMap(p,ps,lw,"probDistFin2D_C&S",True,realPos,True)#Визуализация распределения вероятностей и реального положения