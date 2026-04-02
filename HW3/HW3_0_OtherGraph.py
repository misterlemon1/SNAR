from HW3_BaseFunc import load_data, process_time, latlon_to_meters
import matplotlib.pyplot as plt
import pandas as pd

df = load_data()
process_time(df)
latlon_to_meters(df)

def prepare_additional_data(df):
    cols = [
        'Engine RPM(rpm)',
        ' Altitude',
        'Acceleration Sensor(Total)(g)',
        'Acceleration Sensor(Z axis)(g)'
    ]

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def plot_additional_signals(df):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    signals = [
        ('Engine RPM(rpm)', 'Engine RPM(rpm)'),
        (' Altitude', ' Altitude'),
        ('Acceleration Sensor(Total)(g)','Acceleration Sensor(Total)(g)'),
        ('Acceleration Sensor(Z axis)(g)', 'Acceleration Sensor(Z axis)(g)')
    ]

    for i, (col, label) in enumerate(signals):
        if col in df.columns:
            axs[i].plot(df['t'], df[col])
            axs[i].set_ylabel(label)
            axs[i].grid()
        else:
            axs[i].set_visible(False)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Additional sensor data")

    plt.show()

prepare_additional_data(df)
plot_additional_signals(df)