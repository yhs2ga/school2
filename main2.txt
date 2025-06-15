import streamlit as st
import re
import pickle
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Korpora import Korpora
import os

# -------------------- 유튜브 API 키 --------------------
API_KEY = 'AIzaSyCEPm16vLDOuCxBH7eXB8_c8Kk78kfKfJQ'

# -------------------- 댓글 가져오기 --------------------
def get_comments(video_id, api_key, max_results=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results, 100),
        textFormat="plainText"
    ).execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# -------------------- 유튜브 URL에서 ID 추출 --------------------
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?\\s]|$)", url)
    return match.group(1) if match else None

# -------------------- 모델 학습 또는 불러오기 --------------------
def load_or_train_model():
    model_path = "hate_model.pkl"
    vec_path = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        st.info("처음 실행 중... 모델 학습 중입니다. 1~2분 걸릴 수 있음")

        corpus = Korpora.load("korean_hate_speech")

        # 🔧 구조 자동 판별
        try:
            texts = [sample.text for sample in corpus.train]
            labels = [1 if sample.label in ['hate', 'offensive'] else 0 for sample in corpus.train]
        except AttributeError:
            texts = [text for text, label in corpus.train]
            labels = [1 if label in ['hate', 'offensive'] else 0 for text, label in corpus.train]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)

    return model, vectorizer

# -------------------- 악플 예측 --------------------
def predict_hate(comments, model, vectorizer):
    X = vectorizer.transform(comments)
    preds = model.predict(X)
    hate_comments = [c for c, p in zip(comments, preds) if p == 1]
    return hate_comments

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="유튜브 악플 필터기", layout="centered")
st.title("😡 유튜브 악플 필터기")
st.caption("유튜브 영상 댓글 중 악성 댓글만 골라냅니다")

url = st.text_input("유튜브 영상 URL을 입력하세요:")

if st.button("분석하기"):
    if not API_KEY or "여기에" in API_KEY:
        st.error("유튜브 API 키를 먼저 설정해라 이놈아.")
    elif not url:
        st.warning("URL 안 넣고 버튼 누르면 어쩌자는 거냐?")
    else:
        with st.spinner("댓글 가져오는 중..."):
            video_id = extract_video_id(url)
            if not video_id:
                st.error("영상 ID 추출 실패. URL 제대로 넣었냐?")
            else:
                try:
                    comments = get_comments(video_id, API_KEY, max_results=100)
                except Exception as e:
                    st.error(f"댓글 가져오다 에러남: {e}")
                    comments = []

        if comments:
            with st.spinner("악플 분석 중..."):
                model, vectorizer = load_or_train_model()
                hate_comments = predict_hate(comments, model, vectorizer)

            st.success(f"악플 {len(hate_comments)}개 발견됨")
            if hate_comments:
                for i, hc in enumerate(hate_comments, 1):
                    st.write(f"**{i}.** {hc}")
            else:
                st.info("악플 없음. 세상 아직 살 만하네")
