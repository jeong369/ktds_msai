import streamlit as st
import json
import os
from docx import Document
import zipfile
import requests
import pandas as pd 

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
        f"요구사항: {doc['chunk'][:150]}...\n연관 파트: 화면개발, 모니터링"
        for doc in similar_docs
    ])

    full_prompt = f"""
너는 소프트웨어 개발팀의 AI 도우미야. 고객 요구사항을 분석해서 아래 파트들과 얼마나 연관 있는지 0~1 사이 점수로 표현해줘.

요구사항: "{prompt}"

예시:
{examples}

응답 형식(JSON):
{{
  "요금책정": 0.0,
  "수납": 0.0,
  "화면개발": 0.0,
  "모니터링": 0.0,
  "장애대응": 0.0
}}
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
            {"role": "system", "content": "너는 파트별 연관도 분석 전문가야."},
            {"role": "user", "content": full_prompt}
        ],
        model = DEPLOYMENT_NAME, # 배포이름
        extra_body=rag_params
    )

    st.write("GPT response:", response.choices[0].message.content)
    try:
        return eval(response.choices[0].message.content)
    except:
        return {}

# --- Streamlit UI ---
st.set_page_config(page_title="파트 연관도 분석 대시보드", layout="centered")
st.title("📊 고객 요구사항 기반 파트 연관도 분석")

prompt = st.text_area("고객 요구사항을 입력하세요", height=150)

if st.button("분석 시작") and prompt:
    with st.spinner("벡터화 및 유사 문서 검색 중..."):
        embedding = get_embedding(prompt)
        similar_docs = search_similar_docs(embedding)
        st.write("유사 문서 검색 결과:", similar_docs)

    with st.spinner("GPT로 연관도 분석 중..."):
        result = analyze_parts(prompt, similar_docs)

    if result:
        st.success("분석 완료! 결과는 다음과 같습니다:")

        max_part = max(result, key=result.get)
        for part, score in result.items():
            if part == max_part:
                st.markdown(f"<span style='color:red; font-weight:bold'>{part}: {score:.2f}</span>", unsafe_allow_html=True)
            else:
                st.write(f"{part}: {score:.2f}")
                
        df = pd.DataFrame(list(result.items()), columns=["파트", "연관도 점수"])
        st.bar_chart(df.set_index("파트"))
    else:
        st.error("GPT 응답을 분석하지 못했습니다. 프롬프트 또는 설정을 확인하세요.")




# # 환경변수 로드
# conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# container_name = os.getenv("AZURE_BLOB_CONTAINER")

# # Azure BlobServiceClient 생성
# blob_service_client = BlobServiceClient.from_connection_string(conn_str)
# container_client = blob_service_client.get_container_client(container_name)

# # Streamlit UI
# st.markdown('----')
# uploaded_file = st.file_uploader("문서를 업로드하세요 (.docx)", type=["docx"])

# if uploaded_file is not None:
#     try:
#         # 파일 업로드 → Blob Storage
#         blob_client = container_client.get_blob_client(uploaded_file.name)
#         blob_client.upload_blob(uploaded_file.getvalue(), overwrite=True)

#         st.success(f"✅ {uploaded_file.name} 파일이 성공적으로 업로드되었습니다.")
#     except Exception as e:
#         st.error(f"❌ 업로드 실패: {e}")