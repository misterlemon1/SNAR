import numpy as np
import SNAR2_DrawingTools as tools

#Начальные условия

#Параметры мира
N=10 #Длина мира

sensePerMove=2#Число сканов между перемещениями
pHitReal=0.85#Реальная вероятность успешного скана
pHit=0.8 #Сенсор вернет правильное значение (Наше предположение)
pMiss=0.15 #Сенсор вернет неправильное значение (Наше предположение)
pUx={
    0:0.7,
    1:0.15,
    -1:0.15
}#Вероятности отклонения при смещении по оси x
pUy={
    0:0.8,
    1:0.1,
    -1:0.1
}#Вероятности отклонения при смещении по оси y

iterations=20 #Число итераций цикла

#Т.к. движение коммутативно порядок не важен(сначала x, потом y или наоборот дадут один результат), будем двигаться сначала по x, затем по y
def movex(p,u,pU):
    pu=np.zeros(N)
    for du in pU.keys():
        pu[(u+du)%N]=pU[du]
    new_p=np.zeros(p.shape)
    for i in range(p.size):
        new_p[i//N,:]+=p[i//N,i%N]*pu
        pu=np.roll(pu,1)
    return(new_p)
def movey(p,u,pU):
    pu=np.zeros(N)
    for du in pU.keys():
        pu[(u+du)%N]=pU[du]
    new_p=np.zeros(p.shape)
    for i in range(p.size):
        new_p[:,i//N]+=p[i%N,i//N]*pu
        pu=np.roll(pu,1)
    return(new_p)
