import streamlit as st
import requests
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model
@st.cache_resource
def load_model():
    with open("hate_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data['vectorizer'], data['model']

vectorizer, model = load_model()

# YouTube API Key
API_KEY = 'AIzaSyCEPm16vLDOuCxBH7eXB8_c8Kk78kfKfJQ'

# Extract video ID from URL
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})(?:[&?\\s]|$)", url)
    return match.group(1) if match else None

# Get YouTube comments
def get_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100"
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        res = requests.get(url)
        if res.status_code != 200:
            break

        data = res.json()
        for item in data.get("items", []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# Predict hate comment
def classify_comments(comments):
    X = vectorizer.transform(comments)
    y_pred = model.predict(X)
    return [(c, int(label)) for c, label in zip(comments, y_pred)]

# Streamlit UI
st.title("🧹 유튜브 악플 분류기 (Korpora 기반)")
url = st.text_input("YouTube 영상 URL 입력")

if st.button("분석 시작"):
    with st.spinner("댓글 불러오는 중..."):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("유효한 유튜브 URL이 아닙니다.")
        else:
            try:
                comments = get_comments(video_id)
                st.write(f"총 댓글 수: {len(comments)}개")

                results = classify_comments(comments)
                hate_comments = [c for c, label in results if label == 1]

                st.success(f"악플 감지됨: {len(hate_comments)}개")
                for hc in hate_comments:
                    st.write(f"- {hc}")
            except Exception as e:
                st.error(f"에러 발생: {e}")
