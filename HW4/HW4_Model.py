import tkinter as tk
from tkinter import ttk
import numpy as np
import json
import os

class GNSSCalc:

    # Параметры точности
    Max_iter = 200 #Максимум шагов метода
    Accuracy = 1e-12 #Критерий остановки

    # Основные вычисления
    def calculations(self, coordinates, rho, sigma):

        coordinates = np.array(coordinates, dtype=float) #Фиксированные маяки
        sigma = np.maximum(sigma, 1e-12) #Дисперсия шумов

        w = np.diag(1 / (sigma ** 2)) #Весовая матрица

        centroid = np.mean(coordinates, axis=0) #Начальное приблежение
        x = np.array([centroid[0],centroid[1],0.0]) #Ориентируется на координаты маяков

        #Метод наименьших квадратов
        for i in range(self.Max_iter):

            diff = coordinates - x[:2]
            R = np.linalg.norm(diff, axis=1) #Вектор расстояний
            R = np.maximum(R, 1e-12) #Защита от нуля

            H = np.column_stack(( #Матрица Якоби
                -diff[:, 0] / R,
                -diff[:, 1] / R,
                np.ones(len(R))))

            prediction = R + x[2] #Предсказание модели
            residuals = rho - prediction #Невязки

            try:
                N = H.T @ w @ H #Матрица системы

                if np.linalg.cond(N) > 1e8:
                    return None

                lam = 1e-3

                delta = np.linalg.solve(
                    N + lam * np.eye(3), #Регуляризация
                    H.T @ w @ residuals
                )

            except:
                return None

            x += delta #Обновление

            if np.linalg.norm(delta) < self.Accuracy: #Критерий остановки
                break

        prediction = R + x[2]
        residuals = rho - prediction

        return x, H, residuals


class GNSSApp:

    def __init__(self,root):

        self.root=root
        self.root.title("GNSS Моделирование")
        self.root.resizable(False, False)

        self.calculator = GNSSCalc()

        self.coordinates=[]
        self.receiver=None
        self.counted_pos=None
        self.counted_tau = None

        self.cov=None
        self.residuals=None

        self.gdop=None
        self.pdop=None
        self.tdop=None
        self.rms=None

        self.rho=None

        self.scale=0.5
        self.offset=np.array([0.0,0.0])

        self.drag_start=None
        self.cursor_text=None

        self.noise_mode=tk.StringVar(value="global")

        self.build_ui()
        self.load_state()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # Интерфейс
    def build_ui(self):

        control=ttk.Frame(self.root,width=250)
        control.pack(side=tk.LEFT,fill=tk.Y,padx=10,pady=10)
        control.pack_propagate(False)

        ttk.Label(control, text="Добавить маяк (x,y)").grid(row=0, column=0,columnspan=2)

        self.x_entry = ttk.Entry(control)
        self.x_entry.grid(row=1,column=0)

        self.y_entry = ttk.Entry(control)
        self.y_entry.grid(row=1,column = 1)

        ttk.Button(control, text="Добавить", command=self.add_coordinates_manual).grid(row=2, column = 0)
        ttk.Button(control, text="Удалить", command=self.remove_selected_coordinates).grid(row=2, column=1)
        ttk.Button(control, text="Очистить", command=self.clear).grid(row=4, column=0, columnspan=2)

        self.listbox = tk.Listbox(control)
        self.listbox.grid(row=3, column=0,columnspan=2)

        ttk.Label(control, text="Положение приемника (x,y)").grid(row=5, column=0, columnspan=2)

        self.rx_entry = ttk.Entry(control)
        self.rx_entry.grid(row=6, column=0)

        self.ry_entry = ttk.Entry(control)
        self.ry_entry.grid(row=6, column=1)

        ttk.Button(control, text="Задать", command=self.set_receiver_manual).grid(row=7, column=0, columnspan=2)

        ttk.Radiobutton(control,text="Общая σ",variable=self.noise_mode,value="global").grid(row=8, column=0)
        ttk.Radiobutton(control,text="σ по маякам",variable=self.noise_mode,value="individual").grid(row=8, column=1)

        self.noise_entry=ttk.Entry(control)
        self.noise_entry.insert(0,"5")
        self.noise_entry.grid(row=9, column=0)

        self.noise_list_entry=ttk.Entry(control)
        self.noise_list_entry.insert(0,"5,5,5")
        self.noise_list_entry.grid(row=9, column=1)

        ttk.Label(control, text="τ (ошибка часов)").grid(row=10, column=0,columnspan=2)

        self.clock_entry = ttk.Entry(control)
        self.clock_entry.insert(0, "30")
        self.clock_entry.grid(row=11, column=0,columnspan=2)


        tk.Button(control, text="РЕШИТЬ", command=self.call_calculator, bg="#4169E1", fg="white", font=("Arial", 14, "bold"), height=2, width=20, activebackground="#4169E1").grid(row=15, column=0, columnspan=2)

        self.canvas=tk.Canvas(self.root,width=700,height=700,bg="white")
        self.canvas.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

        self.canvas.bind("<Button-1>", self.add_coordinates_click)
        self.canvas.bind("<Button-3>",self.set_receiver_click)

        self.canvas.bind("<MouseWheel>",self.zoom)
        self.canvas.bind("<Button-4>",self.zoom)
        self.canvas.bind("<Button-5>",self.zoom)

        self.canvas.bind("<Button-2>",self.start_drag)
        self.canvas.bind("<B2-Motion>",self.drag)

        self.canvas.bind("<Motion>",self.show_coords)


#Инструменты

    #Сохранение данных при закрытии
    def on_close(self):
        self.save_state()
        self.root.destroy()

    #Сброс данных
    def data_reset(self):
        self.rho=None
        self.counted_pos=None
        self.cov=None
        self.residuals=None
        self.gdop=None
        self.rms=None
        self.counted_tau = None

    def get_noise(self):

        n=len(self.coordinates)

        if self.noise_mode.get()=="global":
            try:
                s=float(self.noise_entry.get())
            except:
                s=5.0
            return np.full(n,s)

        try:
            vals=list(map(float,self.noise_list_entry.get().split(",")))
        except:
            vals=[5]*n

        if len(vals)<n:
            vals += [vals[-1]]*(n-len(vals))

        return np.array(vals[:n])


    def get_tau(self):
        try:
            return float(self.clock_entry.get())
        except:
            return 0.0


    def zoom(self,event):

        factor=1.1 if (event.delta>0 or getattr(event,"num",0)==4) else 0.9

        mx,my=self.screen_to_world(event.x,event.y)

        self.scale*=factor

        self.offset=np.array([
            mx-(event.x-self.canvas.winfo_width()/2)/self.scale,
            my+(event.y-self.canvas.winfo_height()/2)/self.scale
        ])

        self.redraw()


    def start_drag(self,event):
        self.drag_start=(event.x,event.y)


    def drag(self,event):

        dx=event.x-self.drag_start[0]
        dy=event.y-self.drag_start[1]

        self.offset-=np.array([dx/self.scale,-dy/self.scale])
        self.drag_start=(event.x,event.y)

        self.redraw()


    def show_coords(self,event):

        x,y=self.screen_to_world(event.x,event.y)

        if self.cursor_text:
            self.canvas.delete(self.cursor_text)

        self.cursor_text=self.canvas.create_text(
            event.x+10,event.y-10,
            text=f"({x:.1f},{y:.1f})",
            anchor="nw"
        )

#Функции для данных

    def set_receiver_click(self, event):
        self.receiver = self.screen_to_world(event.x, event.y)
        self.data_reset()
        self.refresh()

    def set_receiver_manual(self):
        try:
            x = float(self.rx_entry.get())
            y = float(self.ry_entry.get())

            self.receiver = (x, y)
            self.data_reset()
            self.refresh()
        except:
            pass

    def add_coordinates_click(self, event):
        self.coordinates.append(self.screen_to_world(event.x, event.y))
        self.data_reset()
        self.refresh()

    def add_coordinates_manual(self):
        try:
            x=float(self.x_entry.get())
            y=float(self.y_entry.get())
            self.coordinates.append((x, y))
            self.data_reset()
            self.refresh()
        except:
            pass

    def remove_selected_coordinates(self):
        s=self.listbox.curselection()
        if s:
            del self.coordinates[s[0]]
            self.data_reset()
            self.refresh()

    def clear(self):
        self.coordinates=[]
        self.receiver=None
        self.data_reset()
        self.refresh()

    def update_list(self):
        self.listbox.delete(0,tk.END)
        for i,(x,y) in enumerate(self.coordinates):
            self.listbox.insert(tk.END,f"{i}: ({x:.1f},{y:.1f})")

    def refresh(self):
        self.update_list()
        self.save_state()
        self.redraw()

    def save_state(self):
        data = {
            "beacons": self.coordinates,
            "receiver": self.receiver,
            "tau": self.clock_entry.get(),
            "noise_mode": self.noise_mode.get(),
            "sigma_global": self.noise_entry.get(),
            "sigma_list": self.noise_list_entry.get()
        }

        with open("gnss_state.json", "w") as f:
            json.dump(data, f)

    def load_state(self):
        if not os.path.exists("gnss_state.json"):
            return

        with open("gnss_state.json") as f:
            data = json.load(f)

        self.coordinates = data.get("beacons", [])
        self.receiver = data.get("receiver")

        # --- восстановление τ ---
        tau = data.get("tau", "30")
        self.clock_entry.delete(0, tk.END)
        self.clock_entry.insert(0, tau)

        # --- режим σ ---
        self.noise_mode.set(data.get("noise_mode", "global"))

        # --- σ ---
        self.noise_entry.delete(0, tk.END)
        self.noise_entry.insert(0, data.get("sigma_global", "5"))

        self.noise_list_entry.delete(0, tk.END)
        self.noise_list_entry.insert(0, data.get("sigma_list", "5,5,5"))

        if self.receiver:
            self.rx_entry.delete(0, tk.END)
            self.rx_entry.insert(0, str(self.receiver[0]))

            self.ry_entry.delete(0, tk.END)
            self.ry_entry.insert(0, str(self.receiver[1]))

        self.refresh()

#Моделирование

    def generate_ranges(self):

        b=np.array(self.coordinates)
        r=np.array(self.receiver)

        d=np.linalg.norm(b-r,axis=1)

        sigma=self.get_noise()
        noise=np.random.normal(0,sigma)

        tau = self.get_tau()

        return d + tau + noise


    def call_calculator(self):

        if len(self.coordinates)<3:
            return

        if self.receiver is None:
            return

        rho = self.generate_ranges()
        self.rho = rho

        sig=np.maximum(self.get_noise(), 1e-8)

        result = self.calculator.calculations(self.coordinates, rho, sig)

        if result is None:
            return

        x, H, residuals = result

        self.counted_pos=tuple(x[:2])
        self.counted_tau = x[2]
        self.residuals = residuals

        try:
            W=np.diag(1/(sig**2))
            Q=np.linalg.inv(H.T@W@H)

            sigma_eff=np.mean(sig)
            self.cov=(sigma_eff**2)*Q[:2,:2]

            G=np.linalg.inv(H.T@H)

            self.gdop=np.sqrt(np.trace(G))
            self.pdop=np.sqrt(G[0,0]+G[1,1])
            self.tdop=np.sqrt(G[2,2])
            self.rms=np.sqrt(np.mean(self.residuals**2))

        except:
            self.cov=None

        self.redraw()


#Перевод координат

    def world_to_screen(self,x,y):
        w=self.canvas.winfo_width()
        h=self.canvas.winfo_height()
        return((x-self.offset[0])*self.scale+w/2,
               -(y-self.offset[1])*self.scale+h/2)


    def screen_to_world(self,sx,sy):
        w=self.canvas.winfo_width()
        h=self.canvas.winfo_height()
        return((sx-w/2)/self.scale+self.offset[0],
               -(sy-h/2)/self.scale+self.offset[1])


#Отображение

    def draw_axes(self):

        w=self.canvas.winfo_width()
        h=self.canvas.winfo_height()

        x0,y0=self.world_to_screen(0,0)

        self.canvas.create_line(0,y0,w,y0)
        self.canvas.create_line(x0,0,x0,h)


    def draw_error_ellipse(self):

        if self.cov is None:
            return

        vals,vecs=np.linalg.eigh(self.cov)
        vals=np.maximum(vals,0)

        idx=np.argsort(vals)[::-1]
        vals=vals[idx]
        vecs=vecs[:,idx]

        if np.any(vals<=0):
            return

        a,b=np.sqrt(vals)*2.447

        t=np.linspace(0,2*np.pi,200)

        E=np.array([a*np.cos(t),b*np.sin(t)])
        E=vecs@E

        x0,y0=self.counted_pos

        pts=[]
        for i in range(len(t)):
            sx,sy=self.world_to_screen(x0+E[0,i],y0+E[1,i])
            pts.extend((sx,sy))

        self.canvas.create_line(pts,fill="orange",width=2,smooth=True)


    def redraw(self):

        self.canvas.delete("all")
        self.draw_axes()

        for x,y in self.coordinates:
            sx,sy=self.world_to_screen(x,y)
            self.canvas.create_oval(sx-6,sy-6,sx+6,sy+6,fill="blue")

        if self.receiver:
            sx,sy=self.world_to_screen(*self.receiver)
            self.canvas.create_oval(sx-6,sy-6,sx+6,sy+6,outline="green",width=2)

        if self.counted_pos:
            sx,sy=self.world_to_screen(*self.counted_pos)
            self.canvas.create_line(sx-8,sy-8,sx+8,sy+8,fill="red",width=2)
            self.canvas.create_line(sx-8,sy+8,sx+8,sy-8,fill="red",width=2)

        self.draw_error_ellipse()

        y = 10
        if self.receiver:
            rx, ry = self.receiver
            self.canvas.create_text(
                10, y,
                anchor="nw",
                text=f"Истинная позиция: ({rx:.3f}, {ry:.3f})",
                font=("Arial", 12, "bold")
            )
            y += 22

        if self.counted_pos:
            ex, ey = self.counted_pos
            self.canvas.create_text(
                10, y,
                anchor="nw",
                text=f"Вычисленная позиция: ({ex:.3f}, {ey:.3f})",
                font=("Arial", 12, "bold"),
            )
            y += 22

        if self.gdop is not None:
            for txt in [
                f"PDOP: {self.pdop:.3f}",
                f"GDOP: {self.gdop:.3f}",
                f"TDOP: {self.tdop:.3f}",
                f"Вычисленное τ: {self.counted_tau:.3f}",
                f"RMS: {self.rms:.3f}",
            ]:
                self.canvas.create_text(10,y,anchor="nw",text=txt,font=("Arial",12,"bold"))
                y+=22

root=tk.Tk()
app=GNSSApp(root)
root.mainloop()