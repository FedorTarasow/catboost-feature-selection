"""Генератор учебного датасета: 1000 строк, 100 признаков, кредитный скоринг.

Запуск: python make_dataset.py  ->  data/scoring_1000x100.csv

Что заложено (знает только генератор, ноутбук это находит сам):
* скрытый риск клиента z и ещё четыре скрытых фактора двигают таргет (дефолт, ~18 %);
* ~25 стабильных полезных признаков — зашумлённые копии скрытых факторов,
  часть из них идёт парами с корреляцией > 0.7 (dpd_max_6m / dpd_max_12m и т.п.);
* 6 признаков с высоким PSI, которые мешают: до октября 2025 они несут «азартный» риск w,
  которого нет в стабильных признаках, и поэтому сильнее всех; в последние три месяца их
  распределение уезжает, а связь с таргетом пропадает. Модель на них хорошо выглядит на val и проваливается на OOT;
* 2 признака с высоким PSI, но без связи с таргетом (сменилась маркетинговая кампания,
  версия приложения);
* 5 пустых или константных признаков (нулевая важность);
* остальное — правдоподобный шум, тоже с несколькими коррелированными парами.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2025
N_ROWS = 1000
N_CLIENTS = 700
MONTHS = pd.period_range("2025-01", "2025-12", freq="M")
OOT_FROM = pd.Period("2025-10", freq="M")  # последние три месяца — out-of-time

rng = np.random.default_rng(SEED)


def e(k=1.0):
    return k * rng.standard_normal(N_ROWS)


# ---------- служебные столбцы ----------
client_of_row = np.concatenate([np.arange(N_CLIENTS), rng.integers(0, N_CLIENTS, N_ROWS - N_CLIENTS)])
rng.shuffle(client_of_row)
month_idx = rng.integers(0, len(MONTHS), N_ROWS)
report_month = MONTHS[month_idx]
is_oot = np.asarray(report_month >= OOT_FROM)

# ---------- скрытые факторы ----------
z_client = rng.standard_normal(N_CLIENTS)
z = 0.75 * z_client[client_of_row] + 0.66 * e()   # общий риск
u_income = e()      # доходы (выше — лучше)
u_debt = e()        # долговая нагрузка
u_tenure = e()      # стаж в банке (выше — лучше)
u_bki = e()         # запросы в БКИ
w = e()             # «азартный» риск: двигает таргет только до октября 2025

logit = (-1.45 + 1.45 * z - 0.60 * u_income + 0.55 * u_debt - 0.40 * u_tenure + 0.40 * u_bki
         + np.where(is_oot, 0.0, 1.10 * w))
target = (rng.random(N_ROWS) < 1 / (1 + np.exp(-logit))).astype(int)

F = {}


def cnt(x, scale=1.0, shift=0.0):
    """Неотрицательный счётчик из непрерывной величины."""
    return np.round(np.clip(scale * (x + shift), 0, None)).astype(float)


def money(x, base=50_000, spread=0.6):
    return np.round(base * np.exp(spread * x), -2)


# ---------- стабильные полезные признаки (25) ----------
dpd12 = z + e(0.75)
F["dpd_max_12m"] = cnt(dpd12, 12, 0.6)
F["dpd_max_6m"] = cnt(0.92 * dpd12 + e(0.3), 10, 0.6)          # пара к dpd_max_12m
F["dpd_cnt_12m"] = cnt(z + e(1.0), 1.5, 0.8)
F["prev_delinq_cnt_24m"] = cnt(0.6 * z + e(1.0), 1.2, 0.7)
F["overdraft_days_3m"] = cnt(0.55 * z + e(1.0), 6, 0.5)
F["payment_to_balance"] = np.round(0.35 - 0.08 * (0.5 * z + e(1.0)), 3)
F["card_txn_cnt_1m"] = cnt(-0.35 * z + e(1.0), 9, 2.2)

sal = -u_income  # знак: выше доход — ниже риск, признак хранит сам доход
sal1 = u_income + e(0.45)
F["salary_inflow_1m"] = money(sal1, 70_000, 0.45)
F["salary_inflow_3m"] = money(0.95 * sal1 + e(0.25), 210_000, 0.43)  # пара
bal = 0.6 * u_income - 0.25 * z + e(0.9)
F["balance_avg_3m"] = money(bal, 40_000, 0.7)
F["balance_avg_6m"] = money(0.93 * bal + e(0.3), 38_000, 0.68)        # пара
F["deposit_sum"] = money(0.5 * u_income + e(1.0), 120_000, 0.9)

util = u_debt + e(0.6)
F["utilization_cc"] = np.round(1 / (1 + np.exp(-util)), 3)
F["utilization_cc_max_6m"] = np.round(1 / (1 + np.exp(-(0.95 * util + e(0.3) + 0.3))), 3)  # пара
F["debt_to_income"] = np.round(np.exp(0.35 * (u_debt - 0.6 * u_income + e(0.6))) * 0.35, 3)
F["loan_cnt_active"] = cnt(0.6 * u_debt + e(1.0), 1.1, 1.6)
F["credit_limit_total"] = money(0.4 * u_income + 0.3 * u_debt + e(1.0), 150_000, 0.6)

ten = u_tenure + e(0.5)
F["tenure_months"] = cnt(ten, 22, 2.5)
F["client_age"] = np.clip(np.round(41 + 9 * (0.65 * ten + e(0.8))), 19, 80).astype(float)
F["mobile_app_logins_1m"] = cnt(0.3 * u_tenure - 0.2 * z + e(1.0), 7, 2.0)

bki = u_bki + e(0.55)
F["inquiries_bki_30d"] = cnt(bki, 1.3, 1.0)
F["inquiries_bki_90d"] = cnt(0.93 * bki + e(0.3), 2.6, 1.0)             # пара
F["new_loans_bki_6m"] = cnt(0.6 * u_bki + 0.3 * z + e(1.0), 1.0, 0.8)
F["bki_score"] = np.round(650 - 55 * (0.5 * z + 0.4 * u_bki + e(0.9)))
F["bki_score_prev"] = np.round(0.9 * F["bki_score"] + 65 + 20 * e())     # пара к bki_score

# ---------- признаки с высоким PSI, которые мешают (6) ----------
# До OOT: почти копия риска z. В OOT: сдвиг распределения и потеря связи с таргетом.
harmful = {
    "gambling_spend_3m":       (dict(base=3_000, spread=0.9), 2.2),
    "crypto_transfers_1m":     (dict(base=5_000, spread=0.8), 1.8),
    "hotline_calls_30d":       (None, 2.0),
    "cash_withdrawal_share_1m": (None, 1.6),
    "microloan_apps_90d":      (None, 2.4),
    "night_txn_share_1m":      (None, 1.7),
}
for name, (m, shift) in harmful.items():
    signal = 0.85 * w + 0.45 * z + e(0.35)
    broken = 0.5 * shift + e(1.6)            # в OOT: другое распределение, без связи с риском
    x = np.where(is_oot, broken, signal)
    if m is not None:
        F[name] = money(x, **m)
    elif name.endswith("share_1m"):
        F[name] = np.round(1 / (1 + np.exp(-(x - 1.0))), 3)
    else:
        F[name] = cnt(x, 1.6, 1.2)

# ---------- признаки с высоким PSI без связи с таргетом (2) ----------
F["promo_offer_cnt_1m"] = cnt(np.where(is_oot, 2.5, 0.0) + e(), 1.5, 1.5)
F["app_version_code"] = np.where(is_oot, 600 + rng.integers(0, 40, N_ROWS), 500 + rng.integers(0, 40, N_ROWS)).astype(float)

# ---------- пустые и константные (5) ----------
F["legacy_score_v1"] = np.full(N_ROWS, np.nan)
F["old_limit_2019"] = np.full(N_ROWS, np.nan)
F["branch_closed_flag"] = np.zeros(N_ROWS)
F["currency_code"] = np.full(N_ROWS, 643.0)
F["kyc_version"] = np.full(N_ROWS, 3.0)

# ---------- шум до 100 признаков ----------
cats = ["food", "fuel", "travel", "pharmacy", "clothes", "electronics", "cafe", "taxi", "utilities", "online"]
noise_names = []
for c in cats:
    noise_names += [f"txn_cnt_{c}_1m", f"txn_sum_{c}_1m", f"txn_sum_{c}_3m"]
noise_names += ["atm_visits_3m", "branch_visits_6m", "sms_open_rate", "email_open_rate", "push_click_rate",
                "card_cnt_active", "reissue_cnt_24m", "address_change_cnt_5y", "phone_change_cnt_5y",
                "insurance_products_cnt", "investment_flag", "payroll_project_flag", "cashback_sum_3m",
                "bonus_points_balance", "auto_payments_cnt", "transfers_in_cnt_1m", "transfers_out_cnt_1m",
                "foreign_txn_share", "contactless_share", "weekend_txn_share", "avg_ticket_1m",
                "max_ticket_3m", "distinct_mcc_3m", "distinct_merchants_3m", "family_members_cnt",
                "dependents_cnt", "region_unemployment", "region_avg_salary", "employer_size_code",
                "work_experience_years", "education_code", "marital_code", "car_owner_flag",
                "realty_owner_flag"]
need = 100 - len(F)
noise_names = noise_names[:need]
assert len(noise_names) == need, (len(noise_names), need)

for i, name in enumerate(noise_names):
    if name.startswith("txn_sum_") and name.endswith("_1m"):
        c = name[len("txn_sum_"):-3]
        base = np.log1p(F[f"txn_cnt_{c}_1m"]) + e(0.35)          # сумма ~ числу операций
        F[name] = money(base, 1_500, 0.8)
    elif name.startswith("txn_sum_") and name.endswith("_3m"):
        c = name[len("txn_sum_"):-3]
        F[name] = np.round(F[f"txn_sum_{c}_1m"] * (2.4 + 0.6 * rng.random(N_ROWS)) + 300 * e(), -1)
    elif name.startswith("txn_cnt_"):
        F[name] = cnt(e(), 4, 2.0)
    elif name.endswith(("_share", "_rate")):
        F[name] = np.round(rng.beta(2, 5, N_ROWS), 3)
    elif name.endswith("_flag"):
        F[name] = (rng.random(N_ROWS) < 0.3).astype(float)
    elif name.endswith("_code"):
        F[name] = rng.integers(1, 6, N_ROWS).astype(float)
    elif name.endswith(("_cnt", "_cnt_1m", "_cnt_5y", "_cnt_24m", "_3m", "_6m")):
        F[name] = cnt(e(), 2, 1.2)
    else:
        F[name] = np.round(np.exp(e(0.5)) * 100, 1)

# реалистичные пропуски в части признаков
for name in ["bki_score", "bki_score_prev", "deposit_sum", "work_experience_years", "education_code",
             "insurance_products_cnt", "salary_inflow_3m"]:
    if name in F:
        F[name] = np.where(rng.random(N_ROWS) < 0.08, np.nan, F[name])

assert len(F) == 100, len(F)

df = pd.DataFrame({
    "client_id": 100_000 + client_of_row,
    "report_month": report_month.astype(str),
    "target": target,
})
feats = pd.DataFrame(F)
feats = feats[sorted(feats.columns)]          # перемешиваем порядок, чтобы «вредные» не стояли кучкой
df = pd.concat([df, feats], axis=1).sort_values(["report_month", "client_id"]).reset_index(drop=True)

out = Path(__file__).with_name("data") / "scoring_1000x100.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(out, df.shape, "bad rate =", round(df["target"].mean(), 3),
      "oot rows =", int((df["report_month"] >= str(OOT_FROM)).sum()))
