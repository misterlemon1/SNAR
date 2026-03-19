import numpy as np
import PIL.Image as img
import PIL.ImageDraw as draw
import PIL.ImageFont as font

colors={
    "r":(255,0,0),
    "g":(0,255,0),
    "b":(0,0,255),
    "p":(200,0,255)
}

def hMap(prob,ps,lw,name : str,data : bool,position,contrast=False):
    """
    :param prob: распределение вероятностей
    :param ps: Размер клетки
    :param lw: толщина граничной линии
    :param name: Имя файла
    :param data: Если True - пишет значение на клетке
    :param position: матрица позиции робота
    :param contrast: Переключает карту в более контрастный режим
    :return:
    """
    ft = font.truetype("arial.ttf", size=ps // 2)
    sz = ps + 2 * lw
    wmap = img.new(mode="RGB", size=(sz * prob.shape[1], sz * prob.shape[0]+80), color=(50, 50, 50))
    d=draw.Draw(wmap)
    barsize=sz * prob.shape[1]/100
    koeff=1
    if contrast:
        koeff=np.max(prob)
        prob/=koeff
    for i in range(0,100):
        d.rectangle((int(barsize*i),sz * prob.shape[0]+60,int(barsize*(i+1)),sz * prob.shape[0]+80),(int(2.55*i),0,int(2.55*(100-i))))
    d.text((0, sz*prob.shape[0]+20), "0", fill=(255, 255, 255), font=ft)
    d.text((sz*prob.shape[1]-sz, sz*prob.shape[0]+20), f"{koeff:.2f}", fill=(255, 255, 255), font=ft)
    ft = font.truetype("arial.ttf", size=ps // 4)
    inrow=prob.shape[1]
    for i in range(prob.size):
        x=i%inrow
        y=i//inrow
        p=prob[y,x]
        d.rectangle((x * sz + lw,y*sz + lw, (x + 1) * sz - lw,(y+1)*sz - lw),(int(255*p),0,int(127*(1-p))))
        if position[y,x]==1:
            d.circle((sz*x+sz//2,sz*y+sz//2),sz//4,(0,127,0))
        if data:
            d.text((sz*x+sz//4,sz*y+sz//4),f"{p*koeff:.2f}",fill=(255,255,255),font=ft)
    wmap.save("Visualization\\" + name + "_heatmap.png")
def wMap2D(w,ps,lw,name : str):
    """
    :param w: мир
    :param ps: сторона квадрата мира
    :param lw: ширина граничной линии
    :param name: Имя файла
    """
    sz=ps+2*lw
    wmap=img.new(mode="RGB",size=(sz*w.shape[1],sz*w.shape[0]),color=(50,50,50))
    d = draw.Draw(wmap)
    inrow=w.shape[1]
    for i in range(w.size):
        x=i%inrow
        y=i//inrow
        d.rectangle((x * sz + lw,y*sz + lw, (x + 1) * sz - lw,(y+1)*sz - lw), colors[w[y,x]])
    wmap.save("Visualization\\" + name + "_worldmap2D.png")
def wMap1D(w,ps,lw,name : str):
    """
    :param w: мир
    :param ps: сторона квадрата мира
    :param lw: ширина граничной линии
    :param name: Имя файла
    """
    sz=ps+2*lw
    wmap = img.new(mode="RGB", size=(sz * len(w),sz), color=(50, 50, 50))
    d=draw.Draw(wmap)
    for i in range(len(w)):
        d.rectangle((i*sz+lw,lw,(i+1)*sz-lw,sz-lw),colors[w[i]])
    wmap.save("Visualization\\"+name+"_worldmap1D.png")