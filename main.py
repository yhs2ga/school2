# app.py
import streamlit as st
from modules.youtube_api import get_comments
from modules.sentiment import analyze_comments

st.title("🧠 유튜브 악플 탐지기")

video_url = st.text_input("유튜브 영상 URL을 입력하세요")
if video_url:
    with st.spinner("댓글 수집 중..."):
        comments = get_comments(video_url)
    
    with st.spinner("악플 분석 중..."):
        analyzed = analyze_comments(comments)

    st.success("분석 완료!")

    st.subheader("📌 악성 댓글")
    for row in analyzed:
        if row['label'] == '악플':
            st.write(f"- {row['text']}")

    st.metric("총 댓글 수", len(analyzed))
    st.metric("악플 수", sum(1 for r in analyzed if r['label'] == '악플'))

