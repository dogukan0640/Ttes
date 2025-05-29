
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def draw_performance_plot():
    try:
        df = pd.read_csv("logs/accuracy_log.txt", sep=" - ", header=None, engine="python")
        df.columns = ["timestamp", "accuracy", "samples"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["accuracy"] = df["accuracy"].str.extract(r"Accuracy: ([0-9.]+)").astype(float)

        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["accuracy"], marker='o')
        plt.title("AI Model Doğruluk Trend Grafiği")
        plt.xlabel("Tarih")
        plt.ylabel("Doğruluk")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("logs/accuracy_plot.png")
        print("Grafik oluşturuldu: logs/accuracy_plot.png")

    except Exception as e:
        print("Grafik hatası:", e)

if __name__ == "__main__":
    draw_performance_plot()
