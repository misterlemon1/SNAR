import FUNCTION as fct

directory= "Teplii_Stan - Yasenevo (Obman №2) - Novoyasenevo/2026-06-11_18-56-46.727_Active 10 Pro"
SESSION_LABEL = "Teplii_Stan - Yasenevo (Obman №2) - Novoyasenevo"           # Заголовок графика; если пусто — берётся из метаданных
SAVE_PATH     = ""           # Путь для сохранения PNG; если пусто — показывается окно

fct.Processing(directory,SESSION_LABEL,SAVE_PATH)