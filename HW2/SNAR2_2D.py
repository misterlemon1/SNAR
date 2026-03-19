import numpy as np
import SNAR2_DrawingTools as tools

#Начальные условия

#Параметры мира
Nx=5 #Длина мира
Ny=5 #Ширина мира
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
sensePerMove=5 #Сканов за одно перемещение

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
        return(np.random.choice(["r","g"],p=[0.5,0.5],size=(Ny,Nx)))
    w=[]
    for i in range(Ny):
        dw=[]
        for j in range(Nx):
            dw+=[wdet[i%wdet.shape[0]][j%wdet.shape[1]]]
        w+=[dw]
    return np.array(w)

def sense(world,idx,idy,pHitReal):
    colors=np.array(["g","r"])
    colProb=np.where(colors==world[idy,idx],pHitReal,1-pHitReal)
    return np.random.choice(colors,p=colProb)



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
tools.wMap2D(world,ps,lw,"Map2D")#Визуализация карты
tools.hMap(p,ps,lw,"probDistInit2D",True,realPos)#Визуализация распределения вероятностей и реального положения

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
        c=sense(world, idrx, idry, pHitReal)
        mask=np.where(world==c,pHit,pMiss)
        p*=mask
        p/=np.sum(p)
    if itr%save_period==0:#Сохранение кратных итераций
        tools.hMap(p, ps, lw, f"probDistIter_{itr}_2D", True, realPos, True)
#Итог
tools.hMap(p,ps,lw,"probDistFin2D",True,realPos,True)#Визуализация распределения вероятностей и реального положения