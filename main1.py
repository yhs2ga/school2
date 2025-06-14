import streamlit as st
import requests
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model
@st.cache_resource
def load_model():
    with open("hate_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model['vectorizer'], model['hate_vectors'], model['hate_texts']

vectorizer, hate_vectors, hate_texts = load_model()

# YouTube API Key
API_KEY = 'AIzaSyCEPm16vLDOuCxBH7eXB8_c8Kk78kfKfJQ'

# Extract video ID (지원: watch, youtu.be, embed, shorts 등)
def extract_video_id(url):
    match = re.search(
        r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})(?:[&?\\s]|$)",
        url
    )
    return match.group(1) if match else None

# Get YouTube comments
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
        if res.status_code != 200:
            break

        data = res.json()

        for item in data.get("items", []):
            text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(text)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# 유사도 기반 악플 판단
SIMILARITY_THRESHOLD = 0.75

def is_hate(comment):
    vec = vectorizer.transform([comment])
    sims = cosine_similarity(vec, hate_vectors)
    max_sim = sims.max()
    return max_sim >= SIMILARITY_THRESHOLD, max_sim

# Streamlit UI
st.title("🔥 유튜브 악플 필터링기 (유사도 기반)")
url = st.text_input("YouTube 영상 URL 입력")

if st.button("악플 분석 시작"):
    with st.spinner("댓글 수집 및 분석 중..."):
        vid = extract_video_id(url)
        if not vid:
            st.error("유효한 YouTube 링크가 아님.")
        else:
            try:
                comments = get_comments(vid)
                st.write(f"총 댓글 수: {len(comments)}개")
                results = []
                for c in comments:
                    try:
                        is_h, sim = is_hate(c)
                        if is_h:
                            results.append({"댓글": c, "유사도": round(sim, 3)})
                    except:
                        continue
                if results:
                    st.success(f"악플 감지됨: {len(results)}개")
                    st.dataframe(results)
                else:
                    st.info("악플 없음 👌")
            except Exception as e:
                st.error(f"에러 발생: {e}")
