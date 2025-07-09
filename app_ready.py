import streamlit as st
import json
import os
from docx import Document
import zipfile

import openai
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

# JSON 파일 경로
DATA_PATH = "streamlit_data/ia_acs_documents.json"
WORD_DOCS_DIR = "streamlit_data/ia_word_documents_50"





# 데이터 불러오기
def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

data = load_data()

# IA 문서번호 생성기: 현재 문서 수 기준 +1
def get_next_doc_id():
    return f"{len(data) + 1:03d}"


st.set_page_config(page_title="IA문서 분석 대시보드", layout="wide")


st.title("💬 요구사항 기반 관련 파트 추천")

# OpenAI API 호출 함수
def get_openai_client(messages):
    """
    Azure OpenAI API를 호출하여 응답을 가져오는 함수
    """
    try:
        # RAG 패턴을 적용하기 위한 추가 파라미터 설정 : Azure Search 자동연동
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

        # OpenAI API 클라이언트 생성
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )

        try:
            response = client.chat.completions.create(
                model = DEPLOYMENT_NAME, # 배포이름
                messages = messages,
                # max_tokens=100,
                extra_body=rag_params,
                # temperature=0.    4 # 일반 대화용
            )
        except Exception as e:
            st.error(f"OpenAI API 호출 중 오류 발생 : {e}")
            return f"Error : {e}"
        
        return response
    
    except Exception as e :
        st.error(f"OpenAI API 호출 중 오류 발생 : {e}")
        return f"Error : {e}"


# 채팅 기록의 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 채팅 기록 표시 <- 화면 출력용
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


# 사용자 입력 받기
if search_query := st.chat_input("🔍 IA 문서 기반 질문을 입력하세요"):
    # 사용자 메시지 저장 및 표시
    st.chat_message("user").write(search_query) # <- 화면 출력용
    st.session_state.messages.append({"role": "user", "content": search_query})


    # OpenAI API 호출 : 모델 응답 생성 및 저장 / Azure Search 문서 기반 RAG
    with st.spinner("응답을 기다리는 중..."):
        assistant_response_tmp = get_openai_client(st.session_state.messages)
        print(assistant_response_tmp)

        assistant_response = assistant_response_tmp.choices[0].message.content
    
    # AI 응답 추가
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").write(assistant_response) # <- 화면 출력용

######################################





