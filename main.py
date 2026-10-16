
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
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

LAST_YEAR = 2025
MIN_OBSERVATION_DAYS = 300


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

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= LAST_YEAR].copy()

    # 연도별 평균기온과 관측일 수 계산
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.error(
        "인터넷 연결, 데이터 주소, CSV의 열 이름을 확인해 주세요."
    )
    st.exception(e)
    st.stop()


# ---------------------------------
# 분석 데이터 확인
# ---------------------------------
if len(yearly) < 2:
    st.error(
        "회귀 직선을 만들 데이터가 부족합니다. "
        "조건을 만족하는 연도가 2개 이상 필요합니다."
    )
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

# ---------------------------------
# 전체 기간 회귀
# ---------------------------------
slope, intercept = np.polyfit(years, temps, 1)
correlation = np.corrcoef(years, temps)[0, 1]

# 1년당 기울기를 100년당 변화량으로 환산
rise_100 = slope * 100


# ---------------------------------
# 최근 20년 회귀
# ---------------------------------
# 가장 최근 분석 연도를 기준으로 20년 구간 설정
recent_start = end_year - 19

recent_df = yearly[
    (yearly["연도"] >= recent_start) &
    (yearly["연도"] <= end_year)
].copy()

recent_slope = None
recent_intercept = None
recent_correlation = None

if len(recent_df) >= 2:
    recent_years = recent_df["연도"].to_numpy(dtype=float)
    recent_temps = recent_df["연평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(
        recent_years, recent_temps, 1
    )

    recent_correlation = np.corrcoef(
        recent_years, recent_temps
    )[0, 1]


# ---------------------------------
# 기온 변화 추세 비교
# ---------------------------------
st.subheader("📊 기온 상승 추세 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌍 전체 기간")

    st.markdown(
        f"""
        <div style="
            background-color: #e8f2ff;
            padding: 24px;
            border-radius: 14px;
            text-align: center;
        ">
            <div style="font-size: 18px; color: #234;">
                100년당 기온 변화
            </div>
            <div style="
                font-size: 42px;
                font-weight: bold;
                color: #145da0;
                margin: 10px 0;
            ">
                {rise_100:+.2f} °C
            </div>
            <div style="font-size: 14px; color: #345;">
                {start_year}~{end_year}년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"회귀 분석에 사용한 연도: **{year_count}개**")
    st.write(f"연도와 기온의 상관계수: **{correlation:.3f}**")

with col2:
    st.markdown("### 🔎 최근 20년")

    if recent_slope is not None:
        recent_rise_100 = recent_slope * 100
        actual_recent_start = int(recent_df["연도"].min())
        actual_recent_end = int(recent_df["연도"].max())

        st.markdown(
            f"""
            <div style="
                background-color: #fff0e5;
                padding: 24px;
                border-radius: 14px;
                text-align: center;
            ">
                <div style="font-size: 18px; color: #543;">
                    100년당 기온 변화
                </div>
                <div style="
                    font-size: 42px;
                    font-weight: bold;
                    color: #c45b17;
                    margin: 10px 0;
                ">
                    {recent_rise_100:+.2f} °C
                </div>
                <div style="font-size: 14px; color: #654;">
                    최근 구간: {actual_recent_start}~{actual_recent_end}년
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            f"회귀 분석에 사용한 연도: **{len(recent_df)}개**"
        )
        st.write(
            f"연도와 기온의 상관계수: **{recent_correlation:.3f}**"
        )
    else:
        st.warning(
            "최근 20년 구간에 분석 가능한 연도가 2개 미만입니다."
        )

st.caption(
    "양수는 상승 추세, 음수는 하강 추세를 뜻합니다. "
    "최근 20년 분석은 마지막 분석 연도를 기준으로 20년 구간을 정하며, "
    "관측일 300일 미만인 해는 제외됩니다."
)


# ---------------------------------
# 전체 기간 분석 정보
# ---------------------------------
st.subheader("📋 회귀 직선의 기준")

info1, info2, info3 = st.columns(3)

info1.metric("분석한 연도 수", f"{year_count}개")
info2.metric("시작 연도", f"{start_year}년")
info3.metric("끝 연도", f"{end_year}년")

st.write(
    f"전체 기간 회귀식: "
    f"예상 기온 = {slope:.5f} × 연도 + ({intercept:.2f})"
)


# ---------------------------------
# 예상 기온 선택
# ---------------------------------
st.subheader("🔮 예상 기온 확인")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 예측은 전체 기간 회귀 직선을 기준으로 계산
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
            전체 기간의 선형 회귀 직선을 이용한 추정값
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Plotly 산점도 및 회귀 직선
# ---------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
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

# 전체 기간 회귀 직선
line_years = np.linspace(start_year, end_year, 200)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3, color="red"),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
if recent_slope is not None:
    recent_line_years = np.linspace(
        int(recent_df["연도"].min()),
        int(recent_df["연도"].max()),
        100
    )
    recent_line_temps = (
        recent_slope * recent_line_years + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, color="orange", dash="dash"),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "최근 20년 회귀 추정 기온: %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

# 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[estimated_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14, symbol="star", color="purple"),
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
# 연도별 데이터 표
# ---------------------------------
with st.expander("연도별 분석 데이터 보기"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)

    display_df = display_df.rename(
        columns={
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------
# 안내
# ---------------------------------
st.info(
    "주의: 기온 변화량은 해당 기간의 직선 추세를 100년 기준으로 환산한 값입니다. "
    "실제 100년 동안의 기온 변화량을 직접 관측한 값과는 다를 수 있습니다. "
    "미래 예상 기온은 과거 추세를 연장한 단순 추정이며, "
    "실제 기온과 다를 수 있습니다."
)
