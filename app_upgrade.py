import streamlit as st
import json
import re
import os
from docx import Document
import zipfile
import requests
import pandas as pd 
import matplotlib.pyplot as plt

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
        f"{i+1}. 요구사항: \"{doc['chunk'][:80]}...\"\n   연관 파트: 화면개발, 모니터링"
        for i, doc in enumerate(similar_docs)
    ])

    # GPT에게 보낼 시스템 및 사용자 메시지
    system_msg = "너는 IT 회사의 소프트웨어 운영팀에서 사용하는 AI 분석 도우미야.\n" \
                 "고객이 입력한 요구사항과 유사한 문서를 바탕으로 어떤 파트(업무 영역)와 연관 있는지를 판단해. " \
                 "각 파트별 연관도를 0.0 ~ 1.0 사이 수치로 출력해줘."

    user_msg = f"""
[입력된 고객 요구사항]
요구사항: "{prompt}"

[IA 문서 예시]
{examples}

[분석 방식]
- 각 IA 문서의 내용, 연관 파트를 참고하여 현재 요구사항이 어떤 파트와 관련이 있는지 판단해.
- 기능 영역, UI/UX, 모니터링, 운영 자동화, 수납, 요금 등의 키워드를 활용해서 판단하되, 단순 키워드 일치보다 문맥의 목적과 작업 흐름을 중점으로 판단해.
- 관련된 파트는 여러 개일 수 있으며, 각 파트에 대해 0.0 ~ 1.0 사이 연관도 점수를 소수점 두 자릿수로 표현해줘.
- 반드시 다음 파트 목록 내에서 판단할 것: ["요금정보", "수납", "화면개발", "모니터링", "운영"]

[출력 형식 예시]
{{
  "요금정보": 0.42,
  "수납": 0.15,
  "화면개발": 0.65,
  "모니터링": 0.91,
  "운영": 0.30
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

    # st.write("GPT response:", response.choices[0])
    # print(response.choices[0].message.content)

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
        st.write("유사 문서 검색 결과:", similar_docs) #잠깐 제거

    with st.spinner("GPT로 연관도 분석 중..."):
        result = analyze_parts(prompt, similar_docs)

    if result:
        st.success("분석 완료! 결과는 다음과 같습니다:")

        # print(result)
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
            print("분석 결과:", result)

            # 🎯 JSON 부분 추출
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            json_part = json.loads(json_match.group()) if json_match else {}

            # 📜 설명 텍스트 추출
            text_part = result[json_match.end():].strip() if json_match else result.strip()

            st.write(json_match)
            st.write(json_part)
            st.write(text_part)

            # ✅ 연관도 막대 그래프
            st.subheader("🔢 파트 연관도 시각화")
            df = pd.DataFrame(json_part.items(), columns=["파트", "연관도"]).sort_values(by="연관도", ascending=True)

            fig, ax = plt.subplots()
            bars = ax.barh(df["파트"], df["연관도"])
            for bar in bars:
                if bar.get_width() == max(df["연관도"]):
                    bar.set_color("red")
            ax.set_xlabel("연관도 (0 ~ 1.0)")
            ax.set_title("요구사항에 대한 파트별 연관도")
            st.pyplot(fig)

            # 📝 GPT 설명 텍스트 출력
            st.subheader("🗒️ GPT 분석 설명")
            st.markdown(text_part)


        except Exception as e:
            st.error(f"분석 결과 파싱 중 오류 발생: {e}")

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