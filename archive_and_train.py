import pandas as pd
import os
import lightgbm as lgb
import pickle
from datetime import datetime

# Arşivleme
today = datetime.now().strftime("%Y-%m-%d")
model_count = len([f for f in os.listdir() if f.startswith("model_") and f.endswith(".pkl")])
model_version = f"model_{model_count+1:03d}.pkl"

signals_file = "signals.csv"
history_file = "signals_history.csv"
performance_log = "performance_log.csv"

if os.path.exists(signals_file):
    df = pd.read_csv(signals_file)
    df.dropna(inplace=True)

    if not df.empty:
        # Arşivle
        if os.path.exists(history_file):
            df.to_csv(history_file, mode="a", header=False, index=False)
        else:
            df.to_csv(history_file, index=False)

        # Eğit
        X = df[["fr", "oi_change", "cvd", "atr"]]
        y = df["label"]
        model = lgb.LGBMClassifier()
        model.fit(X, y)

        with open(model_version, "wb") as f:
            pickle.dump(model, f)

        # Performans log
        success_rate = round((y.sum() / len(y)) * 100, 2)
        perf_data = f"{today},{len(df)},{int(y.sum())},{success_rate},{model_version}\n"
        if os.path.exists(performance_log):
            with open(performance_log, "a") as f:
                f.write(perf_data)
        else:
            with open(performance_log, "w") as f:
                f.write("date,total_signals,success_count,accuracy_percent,model_file\n")
                f.write(perf_data)

        print(f"✅ Eğitim tamamlandı. Model: {model_version} | Doğruluk: %{success_rate}")
    else:
        print("⚠️ signals.csv boş. Eğitim yapılmadı.")
else:
    print("❌ signals.csv bulunamadı.")