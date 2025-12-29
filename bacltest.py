import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# 1. 사용자 설정 (이곳을 자유롭게 수정하세요)
# ==========================================
# [설정 1] 포트폴리오 비중 (종목:비중)
USER_PORTFOLIO = "MAGS:15 / AVGO: 4 / GOOGL:4.5 / QLD: 14 / TQQQ: 7 / XLV: 3 / XLF: 1.5 / XLU: 1.5 / NVDA:3" 

# [설정 2] 비교할 벤치마크 (각각 100% 몰빵 시)
BENCHMARKS = ["QQQM", "TQQQ","QLD","MAGS"]

# [설정 3] 매일 투자할 금액 ($)
DAILY_INVESTMENT = 63.0 

# [설정 4] 백테스팅 시작 날짜 (YYYY-MM-DD 형식)
# None으로 두면 데이터가 가능한 가장 옛날부터 시작
START_DATE = "2024-04-01" 
# START_DATE = None 

# ==========================================
# 2. 데이터 다운로드 및 기간 필터링
# ==========================================
def parse_portfolio(input_str):
    portfolio_map = {}
    parts = input_str.split("/")
    total_weight = 0
    
    for part in parts:
        if ":" not in part: continue
        ticker, weight = part.split(":")
        portfolio_map[ticker.strip().upper()] = float(weight.strip())
        total_weight += float(weight.strip())
    
    return {k: v / total_weight for k, v in portfolio_map.items()}, list(portfolio_map.keys())

# 비중 파싱
my_ratios, my_tickers = parse_portfolio(USER_PORTFOLIO)
all_tickers = list(set(my_tickers + BENCHMARKS))

print(f"▶ 분석 대상: {all_tickers}")
print(f"▶ 데이터 다운로드 중... (기간: {START_DATE if START_DATE else 'Max'})")

# 데이터 다운로드
df = yf.download(all_tickers, period="10y", progress=False, auto_adjust=True)

# 종가(Close)만 추출
if isinstance(df.columns, pd.MultiIndex):
    try:
        df_close = df["Close"].copy()
    except KeyError:
        df_close = df.copy()
else:
    df_close = df["Close"].copy()

# [중요] 날짜 필터링 로직
# 1. 사용자가 지정한 START_DATE 이후 데이터만 남김
if START_DATE:
    df_close = df_close[df_close.index >= pd.Timestamp(START_DATE)]

# 2. 모든 종목의 데이터가 존재하는 구간만 남김 (NaN 제거)
# (예: 사용자가 2020년을 원해도 MAGS가 2023년에 상장했으면 2023년부터 시작)
df_close = df_close.dropna()

if df_close.empty:
    print("❌ 오류: 지정한 날짜 이후에 데이터가 없거나, 공통된 거래 기간이 없습니다.")
    exit()

print(f"▶ 실제 시뮬레이션 기간: {df_close.index[0].date()} ~ {df_close.index[-1].date()}")
print(f"▶ 총 거래일수: {len(df_close)}일")

# ==========================================
# 3. DCA(적립식 투자) 시뮬레이션
# ==========================================
results = pd.DataFrame(index=df_close.index)

# 누적 투자 원금 (매일 $100씩 증가)
results["Invested_Capital"] = range(1, len(results) + 1)
results["Invested_Capital"] *= DAILY_INVESTMENT

# (A) 내 포트폴리오 시뮬레이션
my_port_value = pd.Series(0.0, index=df_close.index)
for ticker, ratio in my_ratios.items():
    # 해당 종목에 할당된 매일 투자금
    daily_alloc = DAILY_INVESTMENT * ratio
    # 해당 종목의 누적 주식수 계산
    cum_shares = (daily_alloc / df_close[ticker]).cumsum()
    # 평가금액 합산
    my_port_value += cum_shares * df_close[ticker]

results["My_Portfolio"] = my_port_value

# (B) 벤치마크 시뮬레이션
for bench in BENCHMARKS:
    cum_shares = (DAILY_INVESTMENT / df_close[bench]).cumsum()
    results[f"DCA_{bench}"] = cum_shares * df_close[bench]

# ==========================================
# 4. 결과 리포트 및 시각화
# ==========================================
final_val = results.iloc[-1]
invested = final_val["Invested_Capital"]

# 요약표 생성
summary = []
cols = ["My_Portfolio"] + [f"DCA_{b}" for b in BENCHMARKS]

for col in cols:
    val = final_val[col]
    profit = val - invested
    roi = (profit / invested) * 100
    summary.append({"Strategy": col, "Final($)": val, "Profit($)": profit, "ROI(%)": roi})

df_summary = pd.DataFrame(summary).sort_values("Final($)", ascending=False)
pd.options.display.float_format = '{:,.2f}'.format

print("\n=== 📊 최종 성과 비교 ===")
print(f"총 투자원금: ${invested:,.2f}")
print(df_summary)

# 그래프
plt.figure(figsize=(12, 6))
plt.plot(results.index, results["Invested_Capital"], 'k--', label="Invested Principal", alpha=0.5)
plt.plot(results.index, results["My_Portfolio"], label="My Portfolio", linewidth=2.5, color='blue')

colors = ['green', 'orange', 'red']
for i, bench in enumerate(BENCHMARKS):
    plt.plot(results.index, results[f"DCA_{bench}"], label=f"DCA {bench}", color=colors[i % len(colors)], alpha=0.7)

plt.title(f"DCA Backtest (Start: {df_close.index[0].date()})")
plt.xlabel("Date")
plt.ylabel("Value ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()