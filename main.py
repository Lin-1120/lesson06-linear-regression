
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------
# 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "회귀 직선을 이용해 선택한 연도의 예상 기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)
LAST_YEAR = 2025


# ---------------------------------
# 데이터 불러오기 및 전처리
# ---------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: " + ", ".join(missing)
        )

    # 날짜와 기온을 올바른 자료형으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준 기간: 2025년까지
    df = df[df["날짜"].dt.year <= LAST_YEAR].copy()

    # 연도별 평균기온 및 관측일 수 계산
    yearly = (
        df.groupby(df["날짜"].dt.year)
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
        .rename(columns={"날짜": "연도"})
    )

    yearly = yearly.rename(columns={"날짜": "연도"})

    # groupby의 인덱스 이름을 명시적으로 정리
    if "연도" not in yearly.columns:
        yearly = yearly.rename(
            columns={yearly.columns[0]: "연도"}
        )

    # 관측일 300일 이상인 해만 사용
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error(
        "기온 데이터를 불러오거나 처리하지 못했습니다. "
        "인터넷 연결과 CSV의 열 이름을 확인해 주세요."
    )
    st.exception(e)
    st.stop()


# ---------------------------------
# 데이터 확인
# ---------------------------------
if len(yearly) < 2:
    st.error(
        "회귀 직선을 만들 데이터가 부족합니다. "
        "조건을 만족하는 연도가 2개 이상 필요합니다."
    )
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 선형 회귀: 기온 = 기울기 * 연도 + 절편
slope, intercept = np.polyfit(years, temps, 1)

predicted_temps = slope * years + intercept

# 피어슨 상관계수
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())
year_count = len(yearly)


# ---------------------------------
# 회귀 분석 결과
# ---------------------------------
st.subheader("📊 회귀 분석 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("분석에 사용한 연도 수", f"{year_count:,}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

col4, col5 = st.columns(2)

with col4:
    st.metric("연도와 평균기온의 상관계수", f"{correlation:.3f}")

with col5:
    st.metric("회귀 직선의 기울기", f"{slope:.4f} °C/년")

st.caption(
    f"분석 기준: 2025년까지, 연간 관측일 {300}일 이상인 해. "
    f"회귀 직선은 {start_year}~{end_year}년의 "
    f"{year_count}개 연평균기온 자료로 계산했습니다."
)


# ---------------------------------
# 연도 선택 및 예상 기온
# ---------------------------------
st.subheader("🔮 몇 년도의 기온이 궁금한가요?")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

estimated_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="
        background-color: #eef6ff;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin: 12px 0 24px 0;
    ">
        <div style="font-size: 22px; color: #345;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 52px;
            font-weight: bold;
            color: #1769aa;
            margin-top: 8px;
        ">
            {estimated_temp:.2f} °C
        </div>
        <div style="font-size: 14px; color: #567; margin-top: 8px;">
            과거 연평균기온의 선형 회귀 직선을 연장한 추정값
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Plotly 산점도 + 회귀 직선
# ---------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7, opacity=0.75),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선: 실제 분석 구간만 표시
line_years = np.linspace(start_year, end_year, 200)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3, color="red"),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도의 예상값을 그래프에 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[estimated_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14, symbol="star", color="orange"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# 데이터 표
# ---------------------------------
with st.expander("연도별 분석 데이터 보기"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df = display_df.rename(
        columns={
            "연도": "연도",
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.info(
    "주의: 이 값은 과거의 선형 추세를 연장한 단순 추정입니다. "
    "미래 기온에는 온실가스 배출, 자연 변동, 도시 환경 등 "
    "다양한 요인이 영향을 주므로 실제 기온과 다를 수 있습니다."
)
