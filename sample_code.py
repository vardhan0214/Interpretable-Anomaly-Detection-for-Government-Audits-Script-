import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime
import seaborn as sns


# === STEP 1: Load Data ===
df = pd.read_csv(r"MEDICAL BIO WASTE 11 July 2025.csv")

# === STEP 2: Standardize column names ===
df.rename(columns={
    'Unique Id': 'unique_id',
    'Month, Day, Year of Vouh Vou Date': 'transaction_date',
    'Vouh Vou No (copy)': 'order_no',
    'Treasury': 'treasury',
    'Sub Treasury': 'sub_treasury_name',
    'Deduct Gst': 'deduct_gst',
    'Deduct It': 'deduct_it',
    'Sub Treasury Flag': 'sub_treasury_flag',
    'Difference in Voush Amount': 'diff_voush_amount',
    'Voush Amount': 'voush_amount',
    'Office dummy': 'office'
}, inplace=True)

# === STEP 3: Convert date ===
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d-%b-%y', errors='coerce')
df = df.sort_values(by=['transaction_date', 'order_no']).reset_index(drop=True)

# === STEP 4: Fiscal Year End and Festival Flags ===
def extract_flags(date):
    if pd.isna(date): return pd.Series([0, 0, 0])
    fy_end = datetime.datetime(date.year, 3, 31)
    fy_score = 3 if date == fy_end else (2 if abs((date - fy_end).days) == 1 else (1 if abs((date - fy_end).days) == 2 else 0))
    holi_flag = int(datetime.datetime(date.year, 3, 17) <= date <= datetime.datetime(date.year, 3, 22))
    diwali_flag = int(datetime.datetime(date.year, 11, 10) <= date <= datetime.datetime(date.year, 11, 15))
    return pd.Series([fy_score, holi_flag, diwali_flag])

df[['fiscal_year_score', 'holi_flag', 'diwali_flag']] = df['transaction_date'].apply(extract_flags)

# === STEP 5: Near Election Flag (±15 days) ===
election_dates = [
    datetime.datetime(2020, 2, 26), datetime.datetime(2020, 10, 25),
    datetime.datetime(2021, 2, 26), datetime.datetime(2022, 1, 8),
    datetime.datetime(2023, 10, 9), datetime.datetime(2024, 3, 16),
    datetime.datetime(2025, 2, 15)
]
df['near_election_flag'] = df['transaction_date'].apply(
    lambda d: int(any(abs((d - e).days) <= 15 for e in election_dates)) if pd.notna(d) else 0
)

# === STEP 6: Lag Features (based on order_no) ===
df['lag1_amt'] = df['voush_amount'].shift(1)
df['lag2_amt'] = df['voush_amount'].shift(2)
df['lead1_amt'] = df['voush_amount'].shift(-1)

# === STEP 7: Modified Z-score (only for voush_amount) ===
def modified_z(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    mad = mad if mad > 1e-6 else 1
    return 0.6745 * (series - median) / mad

df['modz_office'] = df.groupby('office')['voush_amount'].transform(modified_z)
df['modz_treasury'] = df.groupby('treasury')['voush_amount'].transform(modified_z)
df['modz_sub_treasury'] = df.groupby('sub_treasury_name')['voush_amount'].transform(modified_z)

# === STEP 8: Benford’s Law ===
def leading_digit(series):
    return series.astype(str).str.replace(r'[^\d.]', '', regex=True).str.lstrip('0').str[0].astype(float)

def benford_flag(sub_df):
    expected = np.log10(1 + 1 / np.arange(1, 10))
    actual = sub_df['leading'].value_counts(normalize=True).reindex(np.arange(1, 10), fill_value=0)
    distance = np.abs(actual - expected).sum()
    return pd.Series([1 if distance > 0.25 else 0] * len(sub_df), index=sub_df.index)

df['leading'] = leading_digit(df['voush_amount'])
df['benford_flag'] = df.groupby('office').apply(benford_flag).reset_index(level=0, drop=True)
df.drop(columns=['leading'], inplace=True)

# === STEP 9: GST & IT Suspicious Flags ===
df['gst_percent'] = 100 * df['deduct_gst'] / (df['voush_amount'] + 1e-6)
df['it_percent'] = 100 * df['deduct_it'] / (df['voush_amount'] + 1e-6)
df['suspicious_gst_flag'] = ((df['voush_amount'] > 250000) & (df['gst_percent'] < 1.5)).astype(int)
df['suspicious_it_flag'] = ((df['voush_amount'] > 100000) & (df['it_percent'] < 0.5)).astype(int)

# === STEP 10: Anomaly Detection ===
features = [
    'voush_amount', 'deduct_gst', 'deduct_it',
    'fiscal_year_score', 'holi_flag', 'diwali_flag',
    'near_election_flag', 'lag1_amt', 'lag2_amt', 'lead1_amt',
    'modz_office', 'modz_treasury', 'modz_sub_treasury',
    'benford_flag', 'suspicious_gst_flag', 'suspicious_it_flag'
]
X = df[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
df['anomaly'] = model.fit_predict(X_scaled)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# === STEP 11: Reason Tagging ===
def tag_reason(row):
    reasons = []
    if row['fiscal_year_score'] == 3 and row['deduct_it'] < 1000:
        reasons.append("FY End + Low IT")
    if row['modz_office'] > 3.5 or row['modz_treasury'] > 3.5 or row['modz_sub_treasury'] > 3.5:
        reasons.append("Extreme Z-Score")
    if row['benford_flag'] == 1:
        reasons.append("Benford Violation")
    if row['suspicious_gst_flag'] == 1:
        reasons.append("Suspicious GST Deduction")
    if row['suspicious_it_flag'] == 1:
        reasons.append("Suspicious IT Deduction")
    if row['near_election_flag'] == 1:
        reasons.append("Near Election")
    return "; ".join(reasons) if reasons else "Model-Based Anomaly"

df['anomaly_reason'] = df.apply(tag_reason, axis=1)

# === STEP 12: Export Output ===
df[df['anomaly'] == 1].to_csv("anomaly_detailed_output.csv", index=False)
df.to_csv("processed_full_data.csv", index=False)
print("\u2705 Output saved to anomaly_detailed_output.csv (anomalies only with reason tagging)")
print("\u2705 Full processed data saved to processed_full_data.csv (including all features)")
# === STEP 13: Plotting anomalies ===
import matplotlib.pyplot as plt

anomalies = df[df['anomaly'] == 1]

plt.figure(figsize=(12, 7))
plt.scatter(df.index, df['voush_amount'], color='lightgray', label='Normal', alpha=0.6)
plt.scatter(anomalies.index, anomalies['voush_amount'], color='red', label='Anomalies')

# Highlight top 5 highest anomalies
top_anomalies = anomalies.sort_values(by='voush_amount', ascending=False).head(5)

# Plot and annotate top anomalies
for i, row in top_anomalies.iterrows():
    plt.scatter(i, row['voush_amount'], color='darkred', s=100)
    plt.annotate(f"ID: {row['unique_id']}\n₹{int(row['voush_amount'])}",
                 (i, row['voush_amount']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue')

plt.title("Anomaly Detection - Voucher Amounts", fontsize=14)
plt.xlabel("Transaction Index")
plt.ylabel("Voucher Amount (₹)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 💾 Save the plot as a PNG image
plt.savefig("anomaly_scatter_plot.png", dpi=300)

plt.show()
monthly_anomalies = df[df['anomaly'] == 1].groupby(df['transaction_date'].dt.to_period('M')).size()

plt.figure(figsize=(12, 6))
monthly_anomalies.plot(kind='bar', color='crimson')
plt.title("Monthly Count of Anomalies")
plt.xlabel("Month")
plt.ylabel("Number of Anomalies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_anomaly_trend.png", dpi=300)
plt.show()
reason_counts = df[df['anomaly'] == 1]['anomaly_reason'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=reason_counts.values, y=reason_counts.index, palette='coolwarm')
plt.title("Top 10 Anomaly Reasons", fontsize=14)
plt.xlabel("Number of Anomalies")
plt.ylabel("Anomaly Reason")
plt.tight_layout()
plt.savefig("anomaly_reasons_barplot.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='suspicious_gst_flag', data=df, palette='Set2')
plt.title("Count of Suspicious GST Flag")
plt.xlabel("Suspicious GST Flag (0=No, 1=Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("suspicious_gst_flag_count.png", dpi=300)
plt.show()
jessaerogae
plt.figure(figsize=(8, 5))
sns.countplot(x='suspicious_it_flag', data=df, palette='Set1')
plt.title("Count of Suspicious IT Flag")
plt.xlabel("Suspicious IT Flag (0=No, 1=Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("suspicious_it_flag_count.png", dpi=300)
plt.show()


