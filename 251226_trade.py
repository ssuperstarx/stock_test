2. 지수 추종 매수/매도 포지션 코드
%matplotlib inline
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 0) 매일 너가 바꾸는 값 (매일 실행 시 여기만 수정)
# ============================================================
PORTFOLIO_VALUE = 10_000          # 전체 포트폴리오 평가금액(달러)
CURRENT_QQQ_WEIGHT = 0.70         # 현재 QQQ 비중(0~1)

# 리밸런싱(실행) 빈도: "D"(매일) / "W-FRI"(주1회 금요일) / "M"(월말)
REBALANCE_FREQ = "W-FRI"

# 현금 수익률 반영 옵션:
# - "zero": 현금 0% 가정(간단)
# - "SHY" : SHY(단기국채 ETF)로 현금 대체
CASH_MODE = "zero"
CASH_TICKER = "SHY"

# 분위수(Quantile) 계산: 최근 N년 롤링 분포 기준
ROLL_YEARS_FOR_QUANTILE = 2

# 데이터/지표 설정
LOOKBACK_YEARS = 10
ZS_WIN = 252

# ============================================================
# 1) 학습 결과 weight + 방향 (너가 공유한 ridge weight 그대로)
# ============================================================
W_FULL = pd.Series({
    "vix_level": 0.0087,
    "small_big": 0.0079,
    "realized_vol20": 0.0033,
    "cyc_def": 0.0023,
    "adx14": 0.0007,
    "vix_term": -0.0044,
    "credit_risk": -0.0147,
    "trend_200": -0.0162
})

# Risk-on 방향 통일(부호)
DIRECTION = {
    "vix_level": -1,
    "vix_term": -1,
    "realized_vol20": -1,
    "credit_risk": +1,
    "cyc_def": +1,
    "small_big": +1,
    "trend_200": +1,
    "adx14": +1
}

# ============================================================
# 2) 분위수 → 목표비중 (급락에도 반응하도록 "상대순위" 기반)
# ============================================================
def quantile_to_weight(q: float) -> float:
    if q <= 0.10:
        return 0.40
    elif q <= 0.25:
        return 0.55
    elif q <= 0.50:
        return 0.70
    elif q <= 0.75:
        return 0.85
    else:
        return 1.00

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

# ============================================================
# 3) 유틸 함수들
# ============================================================
def zscore_rolling(s: pd.Series, win: int = 252) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return (s - m) / sd

def realized_vol(ret: pd.Series, win: int = 20) -> pd.Series:
    return ret.rolling(win).std(ddof=0) * np.sqrt(252)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, win: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(win).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(win).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(win).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.rolling(win).mean()

def is_exec_day(dt: pd.Timestamp, all_days: pd.DatetimeIndex, freq: str) -> bool:
    if freq == "D":
        return True
    if freq == "W-FRI":
        return dt.weekday() == 4  # Fri
    if freq == "M":
        month_days = all_days[all_days.to_period("M") == dt.to_period("M")]
        return dt == month_days.max()
    raise ValueError("REBALANCE_FREQ must be 'D', 'W-FRI', or 'M'")

# ============================================================
# 4) 데이터 로드 (yfinance)
# ============================================================
base_tickers = ["SPY","QQQ","IWM","HYG","LQD","XLY","XLP","^VIX","^VIX3M"]
tickers = base_tickers.copy()
if CASH_MODE == "SHY":
    tickers = list(dict.fromkeys(tickers + [CASH_TICKER]))

START = (pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)).strftime("%Y-%m-%d")
raw = yf.download(tickers, start=START, auto_adjust=True, group_by="column", progress=False)

def get_field(df, ticker, field="Close"):
    if isinstance(df.columns, pd.MultiIndex):
        return df[field][ticker].copy()
    return df[field].copy()

spy_c = get_field(raw, "SPY", "Close")
spy_h = get_field(raw, "SPY", "High")
spy_l = get_field(raw, "SPY", "Low")

qqq_c = get_field(raw, "QQQ", "Close")
iwn_c = get_field(raw, "IWM", "Close")
hyg_c = get_field(raw, "HYG", "Close")
lqd_c = get_field(raw, "LQD", "Close")
xly_c = get_field(raw, "XLY", "Close")
xlp_c = get_field(raw, "XLP", "Close")
vix_c = get_field(raw, "^VIX", "Close")
vix3m = get_field(raw, "^VIX3M", "Close")

if CASH_MODE == "SHY":
    shy_c = get_field(raw, CASH_TICKER, "Close")
else:
    shy_c = None

# 최신 거래일은 QQQ 종가 확정일
days_all = qqq_c.dropna().index
latest_dt = days_all.max()

print("raw last date:", raw.index.max())
print("Latest QQQ trading date:", latest_dt.date())
print("VIX last date:", vix_c.dropna().index.max().date() if len(vix_c.dropna()) else None)
print("VIX3M last date:", vix3m.dropna().index.max().date() if len(vix3m.dropna()) else None)

# ============================================================
# 5) Feature 계산 + Z-score
# ============================================================
feat = pd.DataFrame(index=spy_c.index)
feat["vix_level"] = vix_c
feat["vix_term"] = vix_c / vix3m
feat["realized_vol20"] = realized_vol(spy_c.pct_change(), 20)
feat["credit_risk"] = hyg_c / lqd_c
feat["cyc_def"] = xly_c / xlp_c
feat["small_big"] = iwn_c / spy_c
feat["trend_200"] = spy_c / spy_c.rolling(200).mean() - 1.0
feat["adx14"] = adx(spy_h, spy_l, spy_c, 14)
feat = feat.replace([np.inf, -np.inf], np.nan)

X = pd.DataFrame(index=feat.index)
for c in feat.columns:
    X[c] = DIRECTION[c] * feat[c]

Xz = X.apply(lambda s: zscore_rolling(s, ZS_WIN))

# ============================================================
# 6) 일별 RAI 계산
# ============================================================
days = days_all  # DatetimeIndex

rai_vals, used_vals, missing_vals = [], [], []
for dt in days:
    available, missing = [], []
    if dt in Xz.index:
        for f in W_FULL.index:
            if pd.notna(Xz.loc[dt, f]):
                available.append(f)
            else:
                missing.append(f)
    else:
        missing = list(W_FULL.index)

    if len(available) < 4:
        rai_val = np.nan
    else:
        Wd = W_FULL[available].copy()
        scale = (W_FULL.abs().sum() / Wd.abs().sum())
        Wd *= scale
        rai_val = float((Xz.loc[dt, available] * Wd).sum())

    rai_vals.append(rai_val)
    used_vals.append(len(available))
    missing_vals.append(", ".join(missing))

rai = pd.Series(rai_vals, index=days, name="RAI")
features_used = pd.Series(used_vals, index=days, name="FeaturesUsed")
missing_s = pd.Series(missing_vals, index=days, name="Missing")

# ============================================================
# 7) Rolling 분위수 → 목표비중
# ============================================================
roll_win = int(252 * ROLL_YEARS_FOR_QUANTILE)
q = rai.rolling(roll_win).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
q = q.fillna(rai.rank(pct=True))
target_w_series = q.apply(quantile_to_weight).rename("TargetWeight")

# ============================================================
# 8) 오늘 의사결정 출력
# ============================================================
rai_today = float(rai.loc[latest_dt]) if pd.notna(rai.loc[latest_dt]) else np.nan
q_today = float(q.loc[latest_dt]) if pd.notna(q.loc[latest_dt]) else np.nan
target_today = float(target_w_series.loc[latest_dt]) if pd.notna(target_w_series.loc[latest_dt]) else np.nan

used_today = int(features_used.loc[latest_dt])
missing_today = missing_s.loc[latest_dt]

cur_w = clamp01(CURRENT_QQQ_WEIGHT)
is_today_exec = is_exec_day(latest_dt, days, REBALANCE_FREQ)

print("\n=== Daily Decision (Quantile-based RAI) ===")
print(f"Latest trading date: {latest_dt.date()}")
print(f"RAI: {rai_today:.3f} | Quantile(rolling {ROLL_YEARS_FOR_QUANTILE}y): {q_today:.3f}")
print(f"Target weight (QQQ): {target_today*100:.0f}% | Current weight: {cur_w*100:.0f}%")
print(f"Features used: {used_today}/{len(W_FULL)}")
if missing_today:
    print("Missing today:", missing_today)

if not is_today_exec:
    print(f"Rebalance schedule: {REBALANCE_FREQ} → 오늘은 실행일이 아님(다음 실행일에 반영)")
else:
    if pd.isna(target_today) or used_today < 4:
        print("Action: HOLD (insufficient features)")
    else:
        delta = target_today - cur_w
        dollars = delta * PORTFOLIO_VALUE
        if abs(delta) < 1e-12:
            print("Action: HOLD (already at target)")
        else:
            if delta > 0:
                print(f"Action: +{delta*100:.1f}%p BUY  (~${abs(dollars):,.0f} QQQ 매수)")
            else:
                print(f"Action: -{abs(delta)*100:.1f}%p SELL (~${abs(dollars):,.0f} QQQ 매도)")

# ============================================================
# 9) 최근 20거래일 스냅샷(각 날짜 Action 포함)  ✅ (여기 수정됨)
# ============================================================
snap_days = days[-30:] if len(days) >= 30 else days
snap_target = target_w_series.reindex(snap_days)

prev_w = clamp01(CURRENT_QQQ_WEIGHT)
prev_weights, actions, exec_flags = [], [], []

for dt in snap_days:
    tw = snap_target.loc[dt]
    prev_weights.append(prev_w)

    exec_today = is_exec_day(dt, days, REBALANCE_FREQ)
    exec_flags.append(exec_today)

    if pd.isna(tw):
        actions.append("HOLD (no target)")
        continue

    delta = tw - prev_w

    if exec_today:
        if abs(delta) < 1e-12:
            actions.append("HOLD (at target) [EXEC]")
        else:
            dollars = delta * PORTFOLIO_VALUE
            if delta > 0:
                actions.append(f"+{delta*100:.1f}%p BUY (~${abs(dollars):,.0f}) [EXEC]")
            else:
                actions.append(f"-{abs(delta)*100:.1f}%p SELL (~${abs(dollars):,.0f}) [EXEC]")
            prev_w = float(tw)
    else:
        if abs(delta) < 1e-12:
            actions.append("HOLD [SCHED]")
        else:
            if delta > 0:
                actions.append(f"+{delta*100:.1f}%p BUY [SCHED]")
            else:
                actions.append(f"-{abs(delta)*100:.1f}%p SELL [SCHED]")

tail = pd.DataFrame({
    "QQQ_Close": qqq_c.reindex(snap_days),
    "RAI": rai.reindex(snap_days),
    "Q": q.reindex(snap_days),
    "TargetW": snap_target,
    "PrevW(approx)": prev_weights,
    "ExecDay": exec_flags,
    "Action": actions,
    "FeaturesUsed": features_used.reindex(snap_days),
    "Missing": missing_s.reindex(snap_days),
})

print("\n--- Last 20 trading days snapshot (with Action) ---")
display(tail.round({"QQQ_Close":2,"RAI":3,"Q":3,"TargetW":2,"PrevW(approx)":2}))

# ============================================================
# 10) (옵션) 최근 1년 차트
# ============================================================
plot_days = days[-252:] if len(days) >= 252 else days

plt.figure(figsize=(15,4))
plt.plot(plot_days, rai.reindex(plot_days).values)
plt.title("RAI (last ~1y)")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15,4))
plt.plot(plot_days, target_w_series.reindex(plot_days).values, label="TargetWeight")
plt.title("Target Weight (last ~1y)")
plt.grid(True, alpha=0.3)
plt.show()
