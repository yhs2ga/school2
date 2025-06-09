import streamlit as st
import requests
import re

# API 키 입력 (넌 이거 숨겨야지. 나중에 환경변수로 빼든가 해)
API_KEY = '여기에_너_API_키_넣어라'

# 악플 키워드 리스트 (예시, 너가 원하는 단어로 바꿔)
HATE_KEYWORDS = ['멍청', '죽어', '좆', 'ㅅㅂ', 'ㄲㅈ', '꺼져', '개새', 'ㅄ', '미친', '병신']

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

def filter_hate_comments(comments):
    hate_comments = []
    for c in comments:
        lowered = c.lower()
        if any(keyword in lowered for keyword in HATE_KEYWORDS):
            hate_comments.append(c)
    return hate_comments

# Streamlit UI
st.title("🧹 YouTube 악플 필터링기")
video_url = st.text_input("YouTube 영상 URL을 입력하세요")
if st.button("분석 시작"):
    with st.spinner("댓글 불러오는 중..."):
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("올바른 YouTube URL이 아님.")
        else:
            try:
                comments = get_comments(video_id)
                st.write(f"전체 댓글 수: {len(comments)}개")
                hate_comments = filter_hate_comments(comments)
                st.success(f"악플 수: {len(hate_comments)}개")

                for hc in hate_comments:
                    st.write(f"- {hc}")

            except Exception as e:
                st.error(f"에러 발생: {e}")
