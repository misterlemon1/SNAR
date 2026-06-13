import FUNCTION as fct

directory= "Yasenevo - Novoyasenevo/2026-06-11_19-12-34.865_Active 10 Pro"
SESSION_LABEL = "Yasenevo - Novoyasenevo"           # Заголовок графика; если пусто — берётся из метаданных
SAVE_PATH     = ""           # Путь для сохранения PNG; если пусто — показывается окно

fct.Processing(directory,SESSION_LABEL,SAVE_PATH)