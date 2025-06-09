import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

API_KEY = '여기에_너_API_키_넣어라'

# 모델 준비
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
    return label  # 0: 부정, 1: 중립, 2: 긍정 (이 모델에 따라 다를 수 있음)

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(video_id):
    comments = []
    next_page_token = ''

    while True:
        url = (
            f"https://www.googleapis.com/youtube/v3/commentThreads"
            f"?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100&pageToken={next_page_token}"
        )
        res = requests.get(url)
        data = res.json()

        for item in data.get("items", []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# Streamlit UI
st.title("🧠 YouTube 악플 감성분석기")
video_url = st.text_input("YouTube 영상 URL을 입력하세요")

if st.button("분석 시작"):
    with st.spinner("댓글 불러오는 중..."):
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("올바른 URL이 아님")
        else:
            try:
                comments = get_comments(video_id)
                st.write(f"전체 댓글 수: {len(comments)}개")

                hate_comments = []
                for c in comments:
                    label = classify_comment(c)
                    if label == 0:  # 부정
                        hate_comments.append(c)

                st.success(f"악플로 분류된 댓글 수: {len(hate_comments)}개")

                for hc in hate_comments:
                    st.write(f"- {hc}")

            except Exception as e:
                st.error(f"에러 발생: {e}")
