import streamlit as st
import requests
import re
from Korpora import Korpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. 모델 학습
@st.cache_resource
def train_model():
    corpus = Korpora.load("korean_hate_speech")
    texts = []
    labels = []
    for d in corpus.train:
        try:
            label_list = getattr(d, "labels", None)
            text = getattr(d, "text", None)
            if label_list and text:
                label = label_list[0]
                texts.append(text)
                labels.append(1 if label in ["hate", "offensive"] else 0)
        except:
            continue

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    return vectorizer, clf

vectorizer, model = train_model()

# 2. 유튜브 API 키
API_KEY = 'AIzaSyCEPm16vLDOuCxBH7eXB8_c8Kk78kfKfJQ'

# 3. 유튜브 링크에서 videoId 추출
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})(?:[&?\s]|$)", url)
    return match.group(1) if match else None

# 4. 댓글 가져오기
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

# 5. 분류 함수
def classify_comments(comments):
    X = vectorizer.transform(comments)
    preds = model.predict(X)
    return [(c, int(p)) for c, p in zip(comments, preds)]

# 6. UI
st.title("🧹 유튜브 악플 필터기 (Korpora 기반)1")
url = st.text_input("YouTube 링크 입력")

if st.button("악플 분석 시작"):
    with st.spinner("댓글 수집 + 악플 분류 중..."):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("유효하지 않은 URL입니다.")
        else:
            try:
                comments = get_comments(video_id)
                st.write(f"총 댓글 수: {len(comments)}개")
                results = classify_comments(comments)
                hate_comments = [c for c, label in results if label == 1]
                st.success(f"악플 감지됨: {len(hate_comments)}개")
                for c in hate_comments:
                    st.write(f"- {c}")
            except Exception as e:
                st.error(f"에러 발생: {e}")

