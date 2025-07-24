import itertools
from ultralytics import YOLO
import os
import shutil

# === CONFIGURATION ===
data_yaml = r"C:\Users\zandro\Desktop\ML project\data\data.yaml"
model_arch = "yolov8n.pt"             # or yolov8s.pt, etc.
results = []

# Grid search parameters
lr_values = [0.001, 0.01]
weight_decays = [0.0005, 0.01]
optimizers = ['SGD', 'Adam']
momentums = [0.6, 0.9]
epochs_values = [20, 50]

# Create all combinations
search_space = list(itertools.product(lr_values, weight_decays, optimizers, momentums, epochs_values))

print(f"🔍 Running {len(search_space)} configurations...\n")

for idx, (lr, wd, opt, mom, epochs) in enumerate(search_space):
    run_name = f"lr{lr}_wd{wd}_{opt}_mom{mom}_ep{epochs}"
    print(f"\n[{idx+1}/{len(search_space)}] Training: {run_name}")

    model = YOLO(model_arch)

    model.train(
        data=data_yaml,
        epochs=epochs,
        lr0=lr,
        optimizer=opt.lower(),  # 'sgd' or 'adam'
        momentum=mom,
        weight_decay=wd,
        project="gridsearch_runs",
        name=run_name,
        verbose=False,
        imgsz=64
    )

    metrics = model.val()
    map_5095 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)

    results.append({
        'name': run_name,
        'lr': lr,
        'weight_decay': wd,
        'optimizer': opt,
        'momentum': mom,
        'epochs': epochs,
        'map50_95': map_5095
    })

# Sort by best mAP50-95 descending
results.sort(key=lambda x: x['map50_95'], reverse=True)

print("\n\n🏆 BEST CONFIGURATIONS (Top 5 by mAP50-95):")
for i, r in enumerate(results[:5]):
    print(f"{i+1}. {r['name']} → mAP50-95: {r['map50_95']:.4f}")





