#1. 매일 사는 종목 (TQQQ) 매수/매도 포지션 코드


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# %matplotlib inline
# =========================
# 설정
# =========================
TICKER = "TQQQ"

# 지표
BB_WINDOW = 20
BB_STD = 2.0
MA200_W = 200

# 조건 (Z-score 기준)
ADD_BUY_Z = -1.5     # (MA200 위) Z <= -1.5면 트레이딩 매수(현금풀 사용)
SELL_Z1   =  1.5     # (MA200 위) Z >= +1.5면 트레이딩 일부매도
SELL_Z2   =  2.0     # (MA200 위) Z >= +2.0면 더 강한 일부매도

# 디폴트 코어 DCA
DCA_DOLLARS_PER_DAY = 7.0

# 트레이딩 매매 비율(수량 기준) - 트레이딩 레이어 보유분에서만 매도
SELL1_FRACTION = 0.20
SELL2_FRACTION = 0.40

# 트레이딩 매수 집행 규칙(현금풀에서만)
TRADE_BUY_USE_FRACTION = 0.50  # 신호 발생 시 현금풀의 50% 사용
MIN_TRADE_BUY_DOLLARS = 200.0  # 현금풀이 너무 작으면 패스

# 데이터
PERIOD = "5y"
INTERVAL = "1d"

# 실제 계좌 기준점 (12/19 장 시작 전이면 최신 거래일은 12/18로 잡히는 게 정상)
START_DATE = "2025-12-18"
INITIAL_EQUITY = 223.57       # 2025-12-18 기준 평가금액
INITIAL_TRADE_CASH = 0.0
INITIAL_TRADE_SHARES = 0.0

# 그래프 표시용 룩백(앵커일 이전 N 거래일을 함께 보여줌)
LOOKBACK_BARS = 252  # 1년 거래일 정도 (원하면 180 등으로 조정)

# =========================
# 데이터 로드
# =========================
df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True,
    group_by="column",
    progress=False,
).dropna()

# 멀티인덱스 컬럼 평탄화(환경별로 필요)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if "Close" not in df.columns:
    raise ValueError(f"'Close' 컬럼이 없습니다. columns={df.columns}")

# =========================
# 지표 계산
# =========================
close = df["Close"].astype(float)

df["MA20"]  = close.rolling(BB_WINDOW).mean()
df["STD20"] = close.rolling(BB_WINDOW).std(ddof=0).replace(0, np.nan)
df["Upper"] = df["MA20"] + BB_STD * df["STD20"]
df["Lower"] = df["MA20"] - BB_STD * df["STD20"]
df["MA200"] = close.rolling(MA200_W).mean()
df["Z"]     = (close - df["MA20"]) / df["STD20"]

df = df.dropna().copy()

if len(df) < MA200_W + BB_WINDOW + 5:
    raise ValueError("데이터가 부족합니다. PERIOD를 더 늘려보세요(예: 10y).")

# =========================
# START_DATE 기준 앵커 거래일 잡기 (START_DATE가 휴장일일 수 있음)
# =========================
start_ts = pd.Timestamp(START_DATE)
df_before_or_on = df[df.index <= start_ts]
if df_before_or_on.empty:
    raise ValueError("START_DATE 이전/당일 데이터가 없습니다. START_DATE/PERIOD를 확인하세요.")

anchor_dt = df_before_or_on.index[-1]  # START_DATE 이전/당일의 마지막 거래일
anchor_price = float(df.loc[anchor_dt, "Close"])

# 시뮬 구간: 앵커일부터
df_sim = df[df.index >= anchor_dt].copy()
if df_sim.empty:
    raise ValueError("df_sim이 비었습니다. date filter를 확인하세요.")

# =========================
# 초기 포지션 주입 (네 실제 계좌를 앵커일 종가로 주식 수 역산)
# =========================
core_shares = INITIAL_EQUITY / anchor_price
trade_shares = float(INITIAL_TRADE_SHARES)
trade_cash = float(INITIAL_TRADE_CASH)

# 신호 계산(시뮬 구간)
df_sim["ADD_BUY"] = (df_sim["Close"] > df_sim["MA200"]) & (df_sim["Z"] <= ADD_BUY_Z)
df_sim["SELL_1"]  = (df_sim["Close"] > df_sim["MA200"]) & (df_sim["Z"] >= SELL_Z1)
df_sim["SELL_2"]  = (df_sim["Close"] > df_sim["MA200"]) & (df_sim["Z"] >= SELL_Z2)

# =========================
# 시뮬레이션 (앵커일은 초기화한 날이라 DCA 스킵, 다음 거래일부터 DCA)
# =========================
records = []

for i, (dt, row) in enumerate(df_sim.iterrows()):
    price = float(row["Close"])

    # 코어 DCA (앵커일 제외)
    core_buy_shares = 0.0
    if i > 0:
        core_buy_shares = DCA_DOLLARS_PER_DAY / price
        core_shares += core_buy_shares

    # 트레이딩 매도 우선(SELL_2 > SELL_1)
    did_trade_sell_shares = 0.0
    did_trade_sell_dollars = 0.0

    if bool(row["SELL_2"]) and trade_shares > 0:
        sell_shares = min(trade_shares, trade_shares * SELL2_FRACTION)
        trade_shares -= sell_shares
        proceeds = sell_shares * price
        trade_cash += proceeds
        did_trade_sell_shares = sell_shares
        did_trade_sell_dollars = proceeds

    elif bool(row["SELL_1"]) and trade_shares > 0:
        sell_shares = min(trade_shares, trade_shares * SELL1_FRACTION)
        trade_shares -= sell_shares
        proceeds = sell_shares * price
        trade_cash += proceeds
        did_trade_sell_shares = sell_shares
        did_trade_sell_dollars = proceeds

    # 트레이딩 매수(현금풀 일부 사용)
    did_trade_buy_shares = 0.0
    did_trade_buy_dollars = 0.0

    if bool(row["ADD_BUY"]) and trade_cash >= MIN_TRADE_BUY_DOLLARS:
        spend = trade_cash * TRADE_BUY_USE_FRACTION
        if spend >= MIN_TRADE_BUY_DOLLARS:
            buy_shares = spend / price
            trade_shares += buy_shares
            trade_cash -= spend
            did_trade_buy_shares = buy_shares
            did_trade_buy_dollars = spend

    # 평가
    core_value = core_shares * price
    trade_value = trade_shares * price
    total_value = core_value + trade_value + trade_cash

    records.append({
        "Date": dt,
        "Close": price,
        "Z": float(row["Z"]),
        "CoreShares": core_shares,
        "TradeShares": trade_shares,
        "TradeCash": trade_cash,
        "CoreValue": core_value,
        "TradeValue": trade_value,
        "TotalValue": total_value,
        "CoreBuyShares": core_buy_shares,
        "TradeBuyShares": did_trade_buy_shares,
        "TradeBuyDollars": did_trade_buy_dollars,
        "TradeSellShares": did_trade_sell_shares,
        "TradeSellDollars": did_trade_sell_dollars,
        "Signal_ADD_BUY": bool(row["ADD_BUY"]),
        "Signal_SELL_1": bool(row["SELL_1"]),
        "Signal_SELL_2": bool(row["SELL_2"]),
    })

bt = pd.DataFrame(records).set_index("Date")

# =========================
# 차트 표시용 구간 확장(앵커일 이전 LOOKBACK_BARS 만큼 보여주기)
# =========================
anchor_pos = df.index.get_loc(anchor_dt)
start_pos = max(0, anchor_pos - LOOKBACK_BARS)
end_pos = df.index.get_loc(df_sim.index[-1]) + 1
df_plot = df.iloc[start_pos:end_pos].copy()

print("df_plot rows =", len(df_plot),
      "| range:", df_plot.index.min().date(), "~", df_plot.index.max().date())
print("df_sim rows =", len(df_sim),
      "| range:", df_sim.index.min().date(), "~", df_sim.index.max().date())
print("anchor_dt:", anchor_dt.date(), "anchor_price:", round(anchor_price, 2))

# =========================
# 차트 표시 (가격 + 밴드 + 신호)
# =========================
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(df_plot.index, df_plot["Close"], label="Close")
ax.plot(df_plot.index, df_plot["MA20"], label="MA20")
ax.plot(df_plot.index, df_plot["Upper"], label="BB Upper (20,2σ)")
ax.plot(df_plot.index, df_plot["Lower"], label="BB Lower (20,2σ)")
ax.plot(df_plot.index, df_plot["MA200"], label="MA200")

# DCA 마커 (시뮬 구간에서 실제로 실행된 날만)
dca_pts = bt[bt["CoreBuyShares"] > 0]
ax.scatter(dca_pts.index, dca_pts["Close"], marker="o", s=25, alpha=0.6, label="Core DCA $100 daily")

# 실제 체결된 트레이딩 매수/매도 표시
trade_buys = bt[bt["TradeBuyDollars"] > 0]
trade_sells = bt[bt["TradeSellDollars"] > 0]
ax.scatter(trade_buys.index, trade_buys["Close"], marker="^", s=90, label="Trade BUY (cash pool)")
ax.scatter(trade_sells.index, trade_sells["Close"], marker="v", s=90, label="Trade SELL (to cash pool)")

# 앵커일 표시
ax.axvline(anchor_dt, linestyle="--", linewidth=1, label="Anchor (baseline)")

ax.set_title(f"{TICKER} | Baseline equity ${INITIAL_EQUITY} on {START_DATE} (anchor {anchor_dt.date()})")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# =========================
# 최신 상태 / 오늘 액션 가이드
# =========================
last_dt = df_sim.index[-1]
last_sig = df_sim.iloc[-1]
last_bt = bt.iloc[-1]

print("\n=== Anchor (your real baseline) ===")
print("Requested START_DATE:", START_DATE)
print("Anchor trading date used:", anchor_dt.date())
print("Anchor Close:", round(anchor_price, 2))
print("Initial Equity:", INITIAL_EQUITY)
print("Initial CoreShares (equity/anchor_close):", round(INITIAL_EQUITY / anchor_price, 6))

print("\n=== Latest ===")
print("Date:", last_dt.date())
print("Close:", round(float(last_sig["Close"]), 2), "Z:", round(float(last_sig["Z"]), 2))
print("Signals => ADD_BUY:", bool(last_sig["Signal_ADD_BUY"]) if "Signal_ADD_BUY" in last_sig else bool(last_sig["ADD_BUY"]),
      "SELL_1:", bool(last_sig["SELL_1"]), "SELL_2:", bool(last_sig["SELL_2"]))

print("\n=== Portfolio Snapshot ===")
print("CoreShares (DCA, never sell):", round(float(last_bt["CoreShares"]), 6))
print("TradeShares (trade layer):", round(float(last_bt["TradeShares"]), 6))
print("TradeCash (from trade sells): $", round(float(last_bt["TradeCash"]), 2))
print("TotalValue (Core + Trade + Cash): $", round(float(last_bt["TotalValue"]), 2))

if bool(last_sig["SELL_2"]) and last_bt["TradeShares"] > 0:
    print("\nAction: SELL strong (trade layer) - if market open and signal persists.")
elif bool(last_sig["SELL_1"]) and last_bt["TradeShares"] > 0:
    print("\nAction: SELL (trade layer) - if market open and signal persists.")
elif bool(last_sig["ADD_BUY"]) and last_bt["TradeCash"] >= MIN_TRADE_BUY_DOLLARS:
    print("\nAction: BUY (trade layer) using cash pool - if market open and signal persists.")
else:
    print("\nAction: No trade-layer action now (only daily DCA runs).")
