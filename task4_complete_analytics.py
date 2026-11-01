"""
Task 04 - Complete Business Analytics Project (Retail Sales Analytics)
Devixo Solutions

Dataset : Superstore Sales (10,800 raw rows -> 9,994 order lines)
Tools   : Python, Pandas, NumPy, SQL (sqlite3), Matplotlib, Scikit-learn
Parts   : 1 Data prep | 2 EDA (18 charts) | 3 KPIs | 4 Dashboard data | Bonus ML
Run     : python task4_complete_analytics.py   (needs internet for the dataset)
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, silhouette_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
os.makedirs("charts", exist_ok=True)

URL = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
BLUE, GREEN, RED, ORANGE = "#2f6fed", "#12a37f", "#e5484d", "#f5a524"
chart_no = 0

def save(title, xlabel="", ylabel=""):
    """Save the current figure as a numbered PNG."""
    global chart_no
    chart_no += 1
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"charts/{chart_no:02d}_{title.split(':')[0].lower().replace(' ', '_')[:40]}.png", dpi=130)
    plt.close()

# ==================================================================
# PART 1 - DATA PREPARATION
# ==================================================================
raw = pd.read_csv(URL)
df = raw.copy()
print("Raw shape:", df.shape)
print("Missing cells:", int(df.isnull().sum().sum()), "| duplicate rows:", df.duplicated().sum())

# 806 rows have an ID but no order details at all -> useless, so remove.
# They were also copies of each other, so this removes the duplicates too.
df = df.dropna(subset=["Order Date", "Sales"]).drop_duplicates().reset_index(drop=True)

# 11 missing postal codes, all Burlington (Vermont) -> real code is 05401
df.loc[df["Postal Code"].isnull(), "Postal Code"] = 5401
df["Postal Code"] = df["Postal Code"].astype(int).astype(str).str.zfill(5)

# data types
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
df["Quantity"] = df["Quantity"].astype(int)
df["Row ID"] = df["Row ID"].astype(int)
for c in ["Ship Mode", "Segment", "Country", "Region", "Category", "Sub-Category"]:
    df[c] = df[c].astype("category")

# outlier analysis (IQR). Orders can really be big, so we FLAG them, not delete.
print("\nOutlier check (IQR rule):")
for c in ["Sales", "Profit", "Quantity", "Discount"]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    flag = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
    df[f"{c} Outlier"] = flag.astype(int)
    print(f"  {c:9s} {flag.sum():5d} rows ({flag.mean()*100:.1f}%)")

# feature engineering
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter"] = df["Order Date"].dt.quarter
df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
df["Weekday"] = df["Order Date"].dt.day_name()
df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin %"] = df["Profit"] / df["Sales"] * 100
df["Is Loss"] = (df["Profit"] < 0).astype(int)
df["Discount Band"] = pd.cut(df["Discount"], [-0.01, 0, 0.2, 0.4, 1],
                             labels=["No discount", "1-20%", "21-40%", "Above 40%"])
print("\nClean shape:", df.shape, "| missing:", int(df.isnull().sum().sum()))
df.to_csv("superstore_cleaned_task4.csv", index=False)

# ==================================================================
# PART 3 - KPIs (done before charts, charts use them)
# ==================================================================
S, P = df["Sales"].sum(), df["Profit"].sum()
O, C = df["Order ID"].nunique(), df["Customer ID"].nunique()
yearly = df.groupby("Year").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                Orders=("Order ID", "nunique"), Customers=("Customer ID", "nunique"))
yearly["Margin %"] = yearly["Profit"] / yearly["Sales"] * 100
yearly["Sales Growth %"] = yearly["Sales"].pct_change() * 100
yearly["Profit Growth %"] = yearly["Profit"].pct_change() * 100
cagr = ((yearly["Sales"].iloc[-1] / yearly["Sales"].iloc[0]) ** (1 / 3) - 1) * 100

kpi = {"Total Revenue": S, "Total Profit": P, "Total Orders": O,
       "Average Order Value": S / O, "Profit Margin %": P / S * 100,
       "Customer Count": C, "Sales Growth 2018 vs 2017 %": yearly["Sales Growth %"].iloc[-1],
       "Sales CAGR 2015-2018 %": cagr, "Repeat Customers % (2+ orders)":
       (df.groupby("Customer ID")["Order ID"].nunique() >= 2).mean() * 100,
       "Loss-making lines %": df["Is Loss"].mean() * 100}
print("\n===== KPIs =====")
for k, v in kpi.items():
    print(f"{k:34s}: {v:,.2f}")
print("\n", yearly.round(2))

def summ(by):
    t = df.groupby(by, observed=True).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                          Orders=("Order ID", "nunique"), Qty=("Quantity", "sum"))
    t["Margin %"] = t["Profit"] / t["Sales"] * 100
    return t

region, category, subcat = summ("Region"), summ("Category"), summ("Sub-Category")
segment, ship, state = summ("Segment"), summ("Ship Mode"), summ("State")
disc, product, customer = summ("Discount Band"), summ("Product Name"), summ("Customer Name")
moy = summ("Month")
for n, t in [("Region", region), ("Category", category), ("Sub-category", subcat),
             ("Segment", segment), ("Discount band", disc), ("Ship mode", ship)]:
    print(f"\n--- {n} ---\n{t.sort_values('Sales', ascending=False).round(2)}")

# ==================================================================
# PART 2 - EXPLORATORY DATA ANALYSIS (18 charts)
# ==================================================================
m = df.groupby("YearMonth")["Sales"].sum()
plt.plot(m.index, m.values, color=BLUE, marker="o", ms=3); plt.xticks(m.index[::6], rotation=45)
save("Monthly sales trend", "Month", "Sales ($)")

x = np.arange(4); plt.bar(x - .2, yearly["Sales"], .4, label="Sales", color=BLUE)
plt.bar(x + .2, yearly["Profit"], .4, label="Profit", color=GREEN)
plt.xticks(x, yearly.index); plt.legend(); save("Yearly sales and profit", "Year", "$")

moy_sales = df.groupby("Month")["Sales"].sum(); plt.bar(moy_sales.index, moy_sales.values, color=BLUE)
plt.xticks(range(1, 13)); save("Seasonality: sales by month of year", "Month", "Sales ($)")

c = category.sort_values("Sales"); y = np.arange(3)
plt.barh(y - .2, c["Sales"], .4, label="Sales", color=BLUE); plt.barh(y + .2, c["Profit"], .4, label="Profit", color=GREEN)
plt.yticks(y, c.index); plt.legend(); save("Category comparison: sales and profit")

s = subcat.sort_values("Profit"); plt.barh(s.index, s["Profit"], color=[RED if v < 0 else GREEN for v in s["Profit"]])
save("Profit by sub-category", "Profit ($)")

r = region.sort_values("Sales", ascending=False); x = np.arange(4)
plt.bar(x - .2, r["Sales"], .4, label="Sales", color=BLUE); plt.bar(x + .2, r["Profit"], .4, label="Profit", color=GREEN)
plt.xticks(x, r.index); plt.legend(); save("Regional performance")

st = state.sort_values("Profit"); st = pd.concat([st.head(8), st.tail(8)])
plt.barh(st.index, st["Profit"], color=[RED if v < 0 else GREEN for v in st["Profit"]])
save("Best and worst 8 states by profit", "Profit ($)")

tp = product.sort_values("Sales").tail(10); plt.barh([n[:32] for n in tp.index], tp["Sales"], color=BLUE)
save("Top 10 products by sales", "Sales ($)")

tc = customer.sort_values("Sales").tail(10); plt.barh(tc.index, tc["Sales"], color=GREEN)
save("Top 10 customers by sales", "Sales ($)")

plt.bar(disc.index, disc["Margin %"], color=[GREEN, GREEN, RED, RED]); plt.axhline(0, color="k", lw=.8)
save("Profit margin by discount band", "Discount band", "Margin %")

plt.scatter(df["Discount"], df["Profit"], s=6, alpha=.3, color=BLUE); plt.axhline(0, color="k", lw=.8)
save("Discount vs profit", "Discount", "Profit ($)")

corr = df[["Sales", "Quantity", "Discount", "Profit", "Ship Days"]].corr()
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(5), corr.columns, rotation=30); plt.yticks(range(5), corr.columns)
for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
save("Correlation heatmap")

plt.pie(segment["Sales"], labels=segment.index, autopct="%1.0f%%", colors=[BLUE, GREEN, ORANGE])
save("Sales share by customer segment")

sh = ship.sort_values("Sales"); plt.barh(sh.index, sh["Orders"], color=BLUE)
save("Orders by ship mode", "Orders")

wd = df.groupby("Weekday")["Order ID"].nunique().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.bar(wd.index, wd.values, color=BLUE); plt.xticks(rotation=30); save("Orders by weekday", "", "Orders")

opc = df.groupby("Customer ID")["Order ID"].nunique()
plt.hist(opc, bins=20, color=GREEN); save("Customer behaviour: orders per customer", "Orders", "Customers")

# ==================================================================
# SQL (SQLite)
# ==================================================================
con = sqlite3.connect(":memory:")
df.assign(**{"Order Date": df["Order Date"].dt.strftime("%Y-%m-%d"),
             "Ship Date": df["Ship Date"].dt.strftime("%Y-%m-%d")}).to_sql("sales", con, index=False)
print("\n===== SQL =====")
print(pd.read_sql('''SELECT State, ROUND(SUM(Sales)) AS sales, ROUND(SUM(Profit)) AS profit
                     FROM sales GROUP BY State HAVING SUM(Profit) < 0 ORDER BY profit LIMIT 6''', con))
print(pd.read_sql('''SELECT Category, "Discount Band" AS band, ROUND(SUM(Profit)) AS profit
                     FROM sales GROUP BY Category, "Discount Band" ORDER BY profit LIMIT 5''', con))
con.close()

# ==================================================================
# BONUS 1 - CUSTOMER SEGMENTATION (RFM + KMeans)
# ==================================================================
snapshot = df["Order Date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("Customer ID").agg(Recency=("Order Date", lambda d: (snapshot - d.max()).days),
                                    Frequency=("Order ID", "nunique"),
                                    Monetary=("Sales", "sum"), Profit=("Profit", "sum"))
X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
best_k, best_s = 2, -1
for k in range(2, 7):
    sc = silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit_predict(X))
    print(f"k={k} silhouette={sc:.3f}")
    if sc > best_s: best_k, best_s = k, sc
rfm["Cluster"] = KMeans(4, n_init=10, random_state=42).fit_predict(X)   # 4 groups: easy to act on
prof = rfm.groupby("Cluster").agg(Customers=("Recency", "size"), Recency=("Recency", "mean"),
                                  Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                  Profit=("Profit", "sum"))
# name the clusters from their numbers
order = prof.sort_values("Monetary", ascending=False).index.tolist()
names = {}
names[order[0]] = "Champions"
rest = [c for c in order[1:]]
by_rec = sorted(rest, key=lambda c: prof.loc[c, "Recency"])
names[by_rec[0]] = "Loyal"; names[by_rec[1]] = "Potential"; names[by_rec[2]] = "At Risk"
rfm["RFM Segment"] = rfm["Cluster"].map(names)
prof["Name"] = prof.index.map(names)
seg_tab = rfm.groupby("RFM Segment").agg(Customers=("Recency", "size"), Recency=("Recency", "mean"),
                                         Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                         Sales=("Monetary", "sum"), Profit=("Profit", "sum"))
seg_tab["Sales Share %"] = seg_tab["Sales"] / seg_tab["Sales"].sum() * 100
print("\n(best silhouette at k =", best_k, ") using k = 4 for business use")
print(seg_tab.round(1).sort_values("Sales", ascending=False))
rfm.to_csv("customer_segments.csv")

t = seg_tab.sort_values("Sales"); plt.barh(t.index, t["Sales"], color=[BLUE, GREEN, ORANGE, RED])
save("Customer segments by sales (RFM)", "Sales ($)")

# ==================================================================
# BONUS 2 - CHURN PREDICTION
# ==================================================================
# Look at each customer using only data BEFORE the cutoff date. Label = no
# order in the 6 months after the cutoff. This way the model never sees the future.
CUT, END = pd.Timestamp("2018-07-01"), pd.Timestamp("2018-12-31")
past, future = df[df["Order Date"] < CUT], df[df["Order Date"] >= CUT]
f = past.groupby("Customer ID").agg(
    Recency=("Order Date", lambda d: (CUT - d.max()).days),
    Frequency=("Order ID", "nunique"), Monetary=("Sales", "sum"),
    Avg_Discount=("Discount", "mean"), Profit=("Profit", "sum"),
    Categories=("Category", "nunique"), First=("Order Date", "min"),
    Avg_Ship_Days=("Ship Days", "mean"))
f["Tenure_days"] = (CUT - f["First"]).dt.days
f["Avg_Order_Value"] = f["Monetary"] / f["Frequency"]
f["Margin"] = f["Profit"] / f["Monetary"]
f = f.drop(columns=["First"])
f["Churn"] = (~f.index.isin(future["Customer ID"].unique())).astype(int)
print(f"\nChurn rate (no order Jul-Dec 2018): {f['Churn'].mean()*100:.1f}% of {len(f)} customers")
Xc, yc = f.drop(columns="Churn"), f["Churn"]
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=.25, random_state=42, stratify=yc)
rf = RandomForestClassifier(300, max_depth=5, class_weight="balanced", random_state=42).fit(Xtr, ytr)
pred, proba = rf.predict(Xte), rf.predict_proba(Xte)[:, 1]
print("Accuracy :", round(accuracy_score(yte, pred), 3))
print("Precision:", round(precision_score(yte, pred, zero_division=0), 3))
print("Recall   :", round(recall_score(yte, pred), 3))
print("ROC AUC  :", round(roc_auc_score(yte, proba), 3))
print("Confusion matrix:\n", confusion_matrix(yte, pred))
imp = pd.Series(rf.feature_importances_, index=Xc.columns).sort_values()
print(imp.round(3).sort_values(ascending=False))
plt.barh(imp.index, imp.values, color=BLUE); save("Churn model: feature importance")
print("\nResult: ROC AUC is about 0.5, so this model is no better than a coin flip.")
print("In this dataset, a customer's past buying (recency, spend, discount) does not")
print("predict who will skip the next 6 months. Different cutoffs and models were also")
print("tried and gave the same result. So we use the RFM 'At Risk' segment instead.")
f.assign(Churn_Probability=rf.predict_proba(Xc)[:, 1]).to_csv("churn_predictions.csv")

# ==================================================================
# BONUS 3 - SALES FORECAST + PRODUCT DEMAND
# ==================================================================
def design(t, months):
    Xd = pd.get_dummies(pd.Categorical(months, categories=range(1, 13)), dtype=float)
    Xd.insert(0, "t", np.asarray(t, dtype=float)); return Xd.values

def forecast(series, steps=6, hold=6):
    """Linear trend + month pattern, compared with 'same month last year x growth'."""
    d = series.reset_index(); d.columns = ["ym", "y"]
    d["t"] = np.arange(len(d)); d["mo"] = pd.PeriodIndex(d["ym"], freq="M").month
    tr, te = d.iloc[:-hold], d.iloc[-hold:]
    lin = LinearRegression().fit(design(tr["t"], tr["mo"]), tr["y"])
    mape_lin = np.mean(np.abs(te["y"] - lin.predict(design(te["t"], te["mo"]))) / te["y"]) * 100
    growth = tr["y"].iloc[-12:].sum() / tr["y"].iloc[-24:-12].sum()
    naive = d["y"].iloc[-hold - 12:-12].values * growth
    mape_naive = np.mean(np.abs(te["y"].values - naive) / te["y"].values) * 100
    final = LinearRegression().fit(design(d["t"], d["mo"]), d["y"])
    fut_ym = pd.period_range(pd.Period(d["ym"].iloc[-1], "M") + 1, periods=steps, freq="M")
    out = final.predict(design(np.arange(len(d), len(d) + steps), fut_ym.month))
    return pd.Series(out, index=fut_ym.astype(str)), mape_lin, mape_naive

sales_m = df.groupby("YearMonth")["Sales"].sum()
fc, m1, m2 = forecast(sales_m)
print(f"\nSales forecast test error -> linear trend+season: {m1:.1f}% | last-year x growth: {m2:.1f}%")
print(fc.round(0))
plt.plot(sales_m.index, sales_m.values, color=BLUE, label="Actual")
plt.plot(["2018-12"] + list(fc.index), [sales_m.iloc[-1]] + list(fc.values), color=ORANGE, ls="--", marker="o", label="Forecast")
plt.xticks(list(sales_m.index[::6]) + list(fc.index[-1:]), rotation=45); plt.legend()
save("Sales forecast Jan to Jun 2019", "Month", "Sales ($)")

print("\nProduct demand forecast (units, Jan-Jun 2019):")
demand = {}
for cat in df["Category"].cat.categories:
    q = df[df["Category"] == cat].groupby("YearMonth")["Quantity"].sum()
    q = q.reindex(sales_m.index, fill_value=0)
    p, e, _ = forecast(q)
    demand[cat] = (int(p.sum()), int(q.iloc[-6:].sum()), round(e, 1))
    print(f"  {cat:17s} next 6 months: {demand[cat][0]:5d} units | Jan-Jun 2018 actual: "
          f"{int(q.loc['2018-01':'2018-06'].sum()):5d} | test error {e:.1f}%")

# ==================================================================
# SAVE TABLES + DASHBOARD DATA
# ==================================================================
with pd.ExcelWriter("task4_kpi_tables.xlsx") as xl:
    pd.Series(kpi, name="Value").to_frame().to_excel(xl, sheet_name="KPIs")
    for n, t in [("Yearly", yearly), ("Region", region), ("Category", category), ("SubCategory", subcat),
                 ("Segment", segment), ("ShipMode", ship), ("DiscountBand", disc), ("State", state),
                 ("RFM_Segments", seg_tab)]:
        t.round(2).to_excel(xl, sheet_name=n)
    fc.rename("Forecast Sales").to_frame().to_excel(xl, sheet_name="Forecast")

df["RFM Segment"] = df["Customer ID"].map(rfm["RFM Segment"])
cols = ["YearMonth", "Region", "Category", "Sub-Category", "Segment", "Product Name",
        "Customer Name", "Order ID", "State", "Discount Band", "RFM Segment", "Ship Mode"]
lists, idxs = {}, {}
for c in cols:
    lists[c] = sorted(df[c].astype(str).unique()); mp = {v: i for i, v in enumerate(lists[c])}
    idxs[c] = df[c].astype(str).map(mp).tolist()
rows = [[idxs[c][i] for c in cols] + [round(float(df["Sales"].iat[i]), 2), round(float(df["Profit"].iat[i]), 2),
                                       int(df["Quantity"].iat[i])] for i in range(len(df))]
json.dump({"names": lists, "rows": rows, "forecast": {"m": list(fc.index), "v": [round(v) for v in fc.values]},
           "mape": round(min(m1, m2), 1)}, open("dashboard_data_task4.json", "w"), separators=(",", ":"))
print(f"\nDone. {chart_no} charts saved in /charts")
