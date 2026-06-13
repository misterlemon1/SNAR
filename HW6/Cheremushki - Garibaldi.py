import FUNCTION as fct

directory= "Cheremushki - Garibaldi/2026-06-11_16-56-21.805_Active 10 Pro"
SESSION_LABEL = "Cheremushki - Garibaldi"           # Заголовок графика; если пусто — берётся из метаданных
SAVE_PATH     = ""           # Путь для сохранения PNG; если пусто — показывается окно

fct.Processing(directory,SESSION_LABEL,SAVE_PATH)