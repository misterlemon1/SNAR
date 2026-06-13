import FUNCTION as fct

directory= "Escalator/2026-06-11_17-43-30.077_Active 10 Pro"
SESSION_LABEL = "Escalator"           # Заголовок графика; если пусто — берётся из метаданных
SAVE_PATH     = ""           # Путь для сохранения PNG; если пусто — показывается окно

fct.Processing(directory,SESSION_LABEL,SAVE_PATH)

directory= "Escalator/2026-06-11_17-50-21.438_Active 10 Pro"
SESSION_LABEL = "Escalator 2"           # Заголовок графика; если пусто — берётся из метаданных
SAVE_PATH     = ""           # Путь для сохранения PNG; если пусто — показывается окно

fct.Processing(directory,SESSION_LABEL,SAVE_PATH)