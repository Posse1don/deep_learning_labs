import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

archive = np.load("results/predictions.npz")
y_val = archive["validation_y"]
s_val = archive["validation_scores"]

print("Побудова кривих та підрахунок ROC-AUC")
roc_auc = roc_auc_score(y_val, s_val)
print(f"ROC-AUC: {roc_auc:.6f}")

precision, recall, _ = precision_recall_curve(y_val, s_val)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation)')
plt.legend()
plt.grid()
plt.savefig("results/pr_validation.png")
plt.close()

fpr, tpr, _ = roc_curve(y_val, s_val)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation)')
plt.legend()
plt.grid()
plt.savefig("results/roc_validation.png")
plt.close()

desc_idx = np.argsort(s_val)[::-1]
s_sorted = s_val[desc_idx]
y_sorted = y_val[desc_idx]

P = np.sum(y_sorted == 1)
N = np.sum(y_sorted == 0)

tp_cum = np.cumsum(y_sorted == 1)
fp_cum = np.cumsum(y_sorted == 0)

distinct_value_indices = np.where(np.diff(s_sorted) != 0)[0]
valid_indices = np.append(distinct_value_indices, len(s_sorted) - 1)

thresholds = s_sorted[valid_indices]
tps = tp_cum[valid_indices]
fps = fp_cum[valid_indices]

thresholds = np.insert(thresholds, 0, np.inf)
tps = np.insert(tps, 0, 0)
fps = np.insert(fps, 0, 0)

fns = P - tps
tns = N - fps
costs = 10 * fns + fps

df_candidates = pd.DataFrame({
    "Threshold": thresholds,
    "TP": tps,
    "FP": fps,
    "FN": fns,
    "TN": tns,
    "C": costs
})

df_candidates.to_csv("results/validation_candidates.csv", index=False)
print("Таблицю кандидатів збережено в results/validation_candidates.csv")

min_cost = df_candidates["C"].min()
best_candidates = df_candidates[df_candidates["C"] == min_cost]
best_row = best_candidates.loc[best_candidates["Threshold"].idxmax()]
t_star = best_row["Threshold"]

print(f"\nТаблиця на Validation")
row_05 = df_candidates.iloc[(df_candidates["Threshold"] - 0.5).abs().argsort()[:1]]

print(f"Стандартний поріг (0.5): C = {row_05['C'].values[0]}")
print(f"Мінімум вартості (t* = {t_star:.6f}): C = {min_cost}")
print(f"Усі негативні (+inf): C = {df_candidates.iloc[0]['C']}")

print("\n---Перевірка алгоритму на малих наборах---\n")

def fast_search(y_val, s_val):
    desc_idx = np.argsort(s_val)[::-1]
    s_sorted, y_sorted = s_val[desc_idx], y_val[desc_idx]
    P, N = np.sum(y_sorted == 1), np.sum(y_sorted == 0)
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    valid_indices = np.append(np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1)
    
    thresholds = np.insert(s_sorted[valid_indices], 0, np.inf)
    tps = np.insert(tp_cum[valid_indices], 0, 0)
    fps = np.insert(fp_cum[valid_indices], 0, 0)
    fns, tns = P - tps, N - fps
    costs = 10 * fns + fps
    
    df = pd.DataFrame({"Threshold": thresholds, "TP": tps, "FP": fps, "FN": fns, "TN": tns, "C": costs})
    best_row = df[df["C"] == df["C"].min()].sort_values("Threshold", ascending=False).iloc[0]
    return df, best_row["Threshold"], best_row["C"]

def slow_search(y_val, s_val):
    candidates = sorted(list(set(s_val)), reverse=True)
    candidates.insert(0, np.inf)
    rows = []
    for t in candidates:
        preds = (s_val >= t).astype(int)
        TP = np.sum((preds == 1) & (y_val == 1))
        FP = np.sum((preds == 1) & (y_val == 0))
        FN = np.sum((preds == 0) & (y_val == 1))
        TN = np.sum((preds == 0) & (y_val == 0))
        rows.append({"Threshold": t, "TP": TP, "FP": FP, "FN": FN, "TN": TN, "C": 10*FN + FP})
    
    df = pd.DataFrame(rows)
    best_row = df[df["C"] == df["C"].min()].sort_values("Threshold", ascending=False).iloc[0]
    return df, best_row["Threshold"], best_row["C"]

datasets = {
    "A": {
        "s": np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1]),
        "y": np.array([0,   1,   0,   1,   0,   0])
    },
    "B": {
        "s": np.array([0.9, 0.6, 0.3]),
        "y": np.array([0,   0,   0])
    },
    "C*": {
        "s": np.array([0.8] + [0.5]*11 + [0.2]),
        "y": np.array([1, 1] + [0]*10 + [0])
    }
}

for name, data in datasets.items():
    s, y = data["s"], data["y"]
    df_fast, t_fast, c_fast = fast_search(y, s)
    df_slow, t_slow, c_slow = slow_search(y, s)
    
    tables_match = df_fast.equals(df_slow)
    
    print(f"\nНабір {name}:")
    print(f"Збіг таблиць алгоритм швидкий/повільний: {tables_match}")
    print(f"Знайдений t*: {t_fast} (Вартість: {c_fast})")

print("\n---Результати на Test---\n")

y_test = archive["test_y"]
s_test = archive["test_scores"]
amount_test = archive["test_amount"]
ids_test = archive["test_row_ids"]

t_star_val = t_star 

preds_test = (s_test >= t_star_val).astype(int)

TP_te = int(np.sum((preds_test == 1) & (y_test == 1)))
FP_te = int(np.sum((preds_test == 1) & (y_test == 0)))
FN_te = int(np.sum((preds_test == 0) & (y_test == 1)))
TN_te = int(np.sum((preds_test == 0) & (y_test == 0)))

C_te = 10 * FN_te + FP_te
precision_te = TP_te / (TP_te + FP_te) if (TP_te + FP_te) > 0 else 0
recall_te = TP_te / (TP_te + FN_te) if (TP_te + FN_te) > 0 else 0

print(f"Матриця невідповідностей: TP={TP_te}, FP={FP_te}, FN={FN_te}, TN={TN_te}")
print(f"Вартість помилок (C): {C_te}")
print(f"Точність: {precision_te:.4f}")
print(f"Повнота: {recall_te:.4f}")

fp_mask = (preds_test == 1) & (y_test == 0)
fn_mask = (preds_test == 0) & (y_test == 1)

fp_indices_all = np.where(fp_mask)[0]
fn_indices_all = np.where(fn_mask)[0]

fp_row_ids = ids_test[fp_indices_all]
fn_row_ids = ids_test[fn_indices_all]

fp_sorted_order = np.argsort(fp_row_ids)
fn_sorted_order = np.argsort(fn_row_ids)

fp_top5_idx = fp_indices_all[fp_sorted_order][:5]
fn_top5_idx = fn_indices_all[fn_sorted_order][:5]

print("\n--- Аналіз помилок ---")
print(f"{'Рядок':<8} | {'Тип':<4} | {'Мітка':<6} | {'Оцінка s':<10} | {'|s - t*|':<10} | {'Amount'}")
print("-" * 65)

for idx in fp_top5_idx:
    dist = abs(s_test[idx] - t_star_val)
    print(f"{ids_test[idx]:<8} | FP   | 0      | {s_test[idx]:<10.6f} | {dist:<10.6f} | {amount_test[idx]}")

for idx in fn_top5_idx:
    dist = abs(s_test[idx] - t_star_val)
    print(f"{ids_test[idx]:<8} | FN   | 1      | {s_test[idx]:<10.6f} | {dist:<10.6f} | {amount_test[idx]}")