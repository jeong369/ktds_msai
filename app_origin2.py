import streamlit as st
import os
from docx import Document
import zipfile
import requests
import pandas as pd 
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

# 프로젝트 내 폰트 경로 <- 웹앱에서 폰트 별도 없기 때문에 matplotlib에 폰트 등록
font_path = os.path.join("font", "NanumGothic-Bold.ttf")
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False
fm.fontManager.addfont(font_path)  # 강제 등록


# import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient


######### openai API를 사용하여 문서 검색으로 변경 함. #########

# 환경변수 로딩
load_dotenv(override=True)

# 환경변수 로딩 시, 
# openai.api_key = os.getenv("AZURE_OPENAI_KEY")
# openai.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# openai.api_type = os.getenv("AZURE_OPENAI_API_TYPE")
# openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_MODEL")

# print(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_TYPE)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

# Azure Cognitive Search 설정
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")

# Azure Cognitive Search 클라이언트 생성
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# 연결 문자열 (Azure Portal > Access keys > Connection string)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")

# OpenAI API 클라이언트 생성
client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)


DATA_PATH = "streamlit_data/ia_docx_parsed.json"
WORD_DOCS_DIR = "streamlit_data/ia_docx"


# --- 공통 함수 ---
def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 팀 목록 불러오기 함수
def extract_teams(data):
    teams = set()
    for doc in data:
        for part in doc.get("parts", []):
            teams.add(part["part_name"])
    return sorted(list(teams))

def get_next_doc_id(data):
    return f"{len(data) + 1:03d}"

# --- 유사 문서 검색 ---
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    # st.write("Embedding response:", response.data[0].embedding)
    return response.data[0].embedding

def search_similar_docs(embedding):
    headers = {
        "Content-Type": "application/json",
        "api-key": SEARCH_API_KEY
    }
    payload = {
        "vector": embedding,
        "top": 5,
        "fields": "text_vector",
        "select": "chunk,title",
        "vectorFields": "text_vector"
    }
    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-preview"
    # url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX_NAME}/docs/search?api-version=2025-01-01-preview"
    response = requests.post(url, headers=headers, json=payload)
    return response.json().get("value", [])


# --- GPT를 통한 파트 연관도 분석 ---
def analyze_parts(prompt, similar_docs):
    examples = "\n".join([
        f"{i+1}. 요구사항: \"{doc['chunk'][:80]}...\"\n   연관 파트: 화면개발, 모니터링"
        for i, doc in enumerate(similar_docs)
    ])

    # GPT에게 보낼 시스템 및 사용자 메시지
    system_msg = "너는 IT 회사의 소프트웨어 운영팀에서 사용하는 AI 분석 도우미야.\n" \
                 "고객이 입력한 요구사항과 유사한 문서를 바탕으로 어떤 파트(업무 영역)와 연관 있는지를 판단해. " \
                 "각 파트별 연관도를 0.0 ~ 1.0 사이 수치로 출력해줘." \
                 "키워드 기반으로 유사한 IA문서도 보여줘"

    user_msg = f"""
[시스템 역할 지시]너는 IT 회사의 소프트웨어 운영팀에서 사용하는 AI 분석 도우미야.
너의 역할은 다음과 같아:
1. 고객이 입력한 요구사항과 유사한 산출물(IA 문서)을 보여주고,
2. 그 문서들과 비교해봤을 때 어떤 파트(업무 영역)가 이 요구사항과 연관되어 있는지를 추론해,
3. 각 파트별 연관도를 수치(0~1.0)로 출력해줘. (1.0에 가까울수록 연관이 높음)

[입력된 고객 요구사항]
요구사항: "{prompt}" ← 사용자의 실제 입력이 들어갈 자리

[IA 문서 예시]
※ 다음은 벡터 유사도로 검색된 실제 IA 산출물들이다. 각 문서에는 관련된 요구사항과 담당 파트가 포함되어 있다.
{examples}

[분석 방식]
- 각 IA 문서의 내용, 연관 파트를 참고하여 현재 요구사항이 어떤 파트와 관련이 있는지 판단해.
- 기능 영역, UI/UX, 모니터링, 운영 자동화, 수납, 요금 등의 키워드를 활용해서 판단하되, 단순 키워드 일치보다 문맥의 목적과 작업 흐름을 중점으로 판단해.
- 관련된 파트는 여러 개일 수 있으며, 각 파트에 대해 0.0 ~ 1.0 사이 연관도 점수를 정수 두 자릿수 소수로 표시해줘. 연관도가 0이어도 무조건 결과에 넣어줘
- 반드시 실제 산출물 문서 내용과 비교하며 판단해.

[출력 형식 예시]
{{
  "요구사항": "",
  "연관파트": [
    {{"파트": "Part A", "연관도": 0.85}},
    {{"파트": "Part C", "연관도": 0.6}}
  ],
  "요약": "",
  "분석 이유" : ""
}}

[제한 사항]
- 반드시 정해진 파트 목록 내에서만 판단해줘: ["요금정보", "수납", "화면개발", "모니터링", "운영"]
- 문서가 없거나 문맥이 애매할 경우, 해당 파트는 생략해도 돼.

"""

    # RAG 패턴을 적용하기 위한 추가 파라미터 설정
    rag_params = {
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": SEARCH_ENDPOINT,
                    "index_name": SEARCH_INDEX_NAME,
                    "authentication": {
                        "type": "api_key",
                        "key": SEARCH_API_KEY
                    },
                    "query_type": "vector",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": EMBEDDING_MODEL,
                    }
                }
            }
        ]
    }

    # st.write("Full prompt for GPT:", full_prompt)

    response = client.chat.completions.create(
        # engine="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        model = DEPLOYMENT_NAME, # 배포이름
        extra_body=rag_params
    )

    #st.write("GPT response:", response.choices[0].message.content)
    # print(response.choices[0].message.content)

    try:
        return eval(response.choices[0].message.content)
    except:
        return {}



# --- Streamlit UI ---
st.set_page_config(page_title="파트 연관도 분석 대시보드", layout="centered")

st.sidebar.title("📂 기능 선택")
mode = st.sidebar.radio("모드를 선택하세요", ["요구사항 분석", "IA 문서기반 조회"])

st.sidebar.markdown("### 📥 문서 다운로드")

# ✅ streamlit_data/ia_docx 내 문서 압축 다운로드
if os.listdir(WORD_DOCS_DIR):
    zip_path = "ia_word_documents_all.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for fname in os.listdir(WORD_DOCS_DIR):
            fpath = os.path.join(WORD_DOCS_DIR, fname)
            zipf.write(fpath, arcname=fname)

    with open(zip_path, "rb") as f:
        st.sidebar.download_button(
            label="📦 전체 문서 압축 다운로드",
            data=f,
            file_name="ia_word_documents_all.zip",
            mime="application/zip"
        )

st.sidebar.markdown("---")

# --- IA 문서 업로드 모드 ---
if mode == "IA 문서기반 조회":
    st.title("📄 IA 문서 업로드 및 저장")

    # 📥 JSON 로드
    @st.cache_data
    def load_json():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    data = load_data()

    # --- 문서 업로드 ---
    uploaded_file = st.file_uploader("IA Word 문서를 업로드하세요", type=["docx"])
    if uploaded_file:
        # 저장
        filename = uploaded_file.name
        save_path = os.path.join(WORD_DOCS_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 파싱
        docx = Document(save_path)
        paragraphs = [p.text.strip() for p in docx.paragraphs if p.text.strip()]
        content = paragraphs

        # 파일명에서 req_id, team 추출
        parts = filename.replace(".docx", "").split("_")
        req_id = parts[0] if len(parts) > 0 else f"REQ{len(data)+1:03d}"
        team = parts[1] if len(parts) > 1 else "Unknown"

        # 새 항목 생성
        new_doc = {
            "req_id": req_id,
            "team": team,
            "filename": filename,
            "content": content
        }

        # 기존 JSON에 추가
        data.append(new_doc)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        st.success(f"✅ 업로드 및 저장 완료: {filename}")


    st.markdown("----")

    # --- 필터링

    # 🧩 팀 목록 생성 (예외 방지)
    teams = sorted(set(d.get("team", "Unknown") for d in data if d.get("team")))
    selected_team = st.selectbox("👨‍💻 팀 선택 (연관 문서 필터링)", ["전체"] + teams)

    # 🔍 키워드 입력
    keyword = st.text_input("🔍 키워드로 문서 내용 필터링 (선택)", placeholder="예: 오류, LAN, 로그 기록 등")

    # 🔎 필터링
    filtered = []
    for d in data:
        team = d.get("team", "")
        if selected_team != "전체" and team != selected_team:
            continue
        if keyword:
            content = d.get("content", [])
            if not any(keyword in para for para in content):
                continue
        filtered.append(d)

    # 📄 과제 선택
    if filtered:
        def safe_title(doc):
            return f"{doc.get('req_id', 'REQXXX')} - {doc.get('filename', 'unknown.docx')}"
        
        req_ids = [safe_title(d) for d in filtered]
        selected_doc = st.selectbox("📄 과제 선택", req_ids)
        selected_id = selected_doc.split(" - ")[0]

        doc = next(
            (d for d in filtered if d.get("req_id") == selected_id and safe_title(d) == selected_doc),
            None
        )

        # 📑 문서 내용 출력
        if doc:
            st.subheader(f"📌 과제 ID: {doc.get('req_id', 'REQXXX')}")
            st.markdown(f"**연관 팀:** `{doc.get('team', 'Unknown')}`")
            st.markdown(f"**파일명:** `{doc.get('filename', '-')}`")
            st.divider()
            st.subheader("📑 문서 내용")

            for i, para in enumerate(doc.get("content", []), 1):
                if keyword and keyword in para:
                    st.markdown(f"✅ {para}")
                else:
                    st.markdown(f"{para}")
        else:
            st.warning("📄 선택된 문서를 찾을 수 없습니다.")
    else:
        st.warning("🔍 조건에 해당하는 문서가 없습니다.")





# --- 요구사항 분석 모드 ---
elif mode == "요구사항 분석":
    st.title("📊 고객 요구사항 기반 파트 연관도 분석")
    prompt = st.text_area("고객 요구사항을 입력하세요", height=150)

    if st.button("분석 시작") and prompt:
        with st.spinner("벡터화 및 유사 문서 검색 중..."):
            embedding = get_embedding(prompt)
            similar_docs = search_similar_docs(embedding)
            # st.write("유사 문서 검색 결과:", similar_docs) --잠깐 제거

        with st.spinner("GPT로 연관도 분석 중..."):
            result = analyze_parts(prompt, similar_docs)

        if result:
            st.success("분석 완료! 결과는 다음과 같습니다:")

            print(result)


            # max_part = max(result, key=result.get)
            # for part, score in result.items():
            #     if part == max_part:
            #         st.markdown(f"<span style='color:red; font-weight:bold'>{part}: {score:.2f}</span>", unsafe_allow_html=True)
            #     else:
            #         st.write(f"{part}: {score:.2f}")
                    
            # df = pd.DataFrame(list(result.items()), columns=["파트", "연관도 점수"])
            # st.bar_chart(df.set_index("파트"))

            
            # 결과 파싱
            try :

                # 문자열이면 json.loads, dict면 그대로 사용
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError as e:
                        st.error(f"❌ GPT 응답 파싱 실패: {e}")
                        result = None
                elif isinstance(result, dict):
                    result = result
                else:
                    st.error("❌ GPT 응답 형식이 예상과 다릅니다.")
                    result = None

            
                # 📝 요약 및 분석 표시
                st.markdown("### 📝 요구사항")
                st.write(result.get("요구사항", "-"))

                st.markdown("### 📌 요약")
                st.success(result.get("요약", "-"))

                st.markdown("### 🧠 분석 이유")
                st.info(result.get("분석 이유", "-"))

                # 📑 유사 문서
                if "유사 IA 문서" in result:
                    st.markdown("### 📄 유사 IA 문서")
                    st.code(result["유사 IA 문서"])

                # 📊 연관도 시각화
                st.markdown("### 📊 파트별 연관도")

                df = pd.DataFrame(result["연관파트"])
                df = df.sort_values("연관도", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.barh(df["파트"], df["연관도"])
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("연관도 (0.0 ~ 1.0)")
                ax.set_title("파트별 연관도 분석 결과")
                st.pyplot(fig)


            except Exception as e:
                st.error(f"분석 결과 파싱 중 오류 발생: {e}")

        else:
            st.error("지금은 GPT가 퇴근했어요. 다음에 다시 시도하세요!")


