import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 앱 제목
st.title("📊 간단한 데이터 시각화 대시보드")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 데이터 미리보기
    st.subheader("📋 데이터 미리보기")
    st.write(df.head())

    # 선택할 컬럼 지정
    numeric_columns = df.select_dtypes(['number']).columns
    selected_column = st.selectbox("📌 분석할 수치형 컬럼을 선택하세요", numeric_columns)

    # 히스토그램 시각화
    st.subheader(f"📊 {selected_column} 분포도")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], kde=True, bins=20, ax=ax)
    st.pyplot(fig)

    # 상관관계 분석
    st.subheader("📌 상관관계 분석")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # 데이터 필터링
    st.subheader("📌 데이터 필터링")
    min_value, max_value = st.slider(f"{selected_column} 값 범위 선택", 
                                     float(df[selected_column].min()), 
                                     float(df[selected_column].max()), 
                                     (float(df[selected_column].min()), float(df[selected_column].max())))
    filtered_df = df[(df[selected_column] >= min_value) & (df[selected_column] <= max_value)]
    st.write(filtered_df)

    # 데이터 다운로드 버튼
    st.download_button("📥 필터링된 데이터 다운로드", 
                       filtered_df.to_csv(index=False).encode('utf-8'), 
                       "filtered_data.csv", 
                       "text/csv")
