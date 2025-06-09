import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re

API_KEY = 'AIzaSyBGiCgfY5Vjyh7j5xoYr__fwb1E1vSBxWA'

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base")
    return tokenizer, model

tokenizer, model = load_model()

def classify_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()
    return label, probs.tolist()[0]

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(video_id):
    comments = []
    next_page_token = None
    while True:
        base_url = (
            f"https://www.googleapis.com/youtube/v3/commentThreads"
            f"?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
        )
        if next_page_token:
            base_url += f"&pageToken={next_page_token}"

        res = requests.get(base_url)
        data = res.json()

        for item in data.get("items", []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# Streamlit UI
st.title("🧠 YouTube 댓글 감성 분석기")
video_url = st.text_input("YouTube 영상 URL 입력")

if st.button("분석 시작"):
    with st.spinner("댓글 분석 중..."):
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("URL이 이상한데?")
        else:
            try:
                comments = get_comments(video_id)
                st.write(f"총 댓글 수: {len(comments)}개")

                results = []
                label_map = {0: "부정", 1: "긍정"}
                status_text = st.empty()
                progress_bar = st.progress(0)

                for idx, c in enumerate(comments):
                    try:
                        label, probs = classify_comment(c)
                        results.append({
                            "댓글": c,
                            "감성": label_map[label],
                            "부정 확률": round(probs[0], 3),
                            "긍정 확률": round(probs[1], 3)
                        })
                    except Exception:
                        continue

                    # 진행률 업데이트
                    percent_complete = int((idx + 1) / len(comments) * 100)
                    progress_bar.progress(percent_complete)
                    status_text.text(f"{idx + 1} / {len(comments)}개 분석 완료")

                df = pd.DataFrame(results)
                st.dataframe(df)

            except Exception as e:
                st.error(f"에러 발생: {e}")
