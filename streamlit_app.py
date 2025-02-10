import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Streamlit 앱 제목
st.title("📊 데이터 대시보드 및 머신러닝 예측")

# 파일 업로드
st.sidebar.header("📂 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 데이터 미리보기")
    st.write(df.head())

    # 데이터 정보 출력
    st.write("### 데이터 정보")
    st.write(df.describe())

    # 컬럼 선택
    st.sidebar.header("📌 데이터 시각화 설정")
    column = st.sidebar.selectbox("시각화할 컬럼 선택", df.columns)

    # 히스토그램 시각화
    st.write(f"### 📊 {column} 컬럼 분포")
    fig, ax = plt.subplots()
    sns.histplot(df[column], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # 상관관계 분석
    st.write("### 🔗 상관관계 분석")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 머신러닝 예측 (선형 회귀)
    st.sidebar.header("🤖 머신러닝 예측")
    target = st.sidebar.selectbox("예측할 타겟 컬럼 선택", df.columns)

    features = st.sidebar.multiselect("사용할 입력 변수 선택", df.columns, default=df.columns[:-1])

    if st.sidebar.button("🔍 예측 실행"):
        X = df[features]
        y = df[target]

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 학습
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 예측 결과
        score = model.score(X_test, y_test)
        st.write(f"### 📈 회귀 모델 성능 (R² Score): {score:.4f}")

        # 실제 값 vs 예측 값 비교 그래프
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("실제 값")
        ax.set_ylabel("예측 값")
        st.pyplot(fig)

else:
    st.write("📂 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")
