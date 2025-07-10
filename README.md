# README

- 계정

  > new ms azure : labuser24@helloaicloud.onmicrosoft.com / !!Seoul2025 > 강사님 개인 tenant.. > 융합대학 실습계정

- 인증키

  ```.env
  AZURE_OPENAI_ENDPOINT=https://user24-openai-005.openai.azure.com/
  AZURE_OPENAI_KEY=aznRrmqonPZ0J7mxagWJC9mRqLyATCp7rIwcn5jcYerAc3hQLm10JQQJ99BGACYeBjFXJ3w3AAABACOG5UKY
  AZURE_OPENAI_API_TYPE=azure
  AZURE_OPENAI_API_VERSION=2024-12-01-preview
  AZURE_OPENAI_CHAT_MODEL=cij-gpt-4.1-002
  
  AZURE_OPENAI_EMBEDDING_MODEL=cij-text-embedding-3-small-002
  
  SEARCH_API_KEY=s8yFItIASLMgNkGBQD2jNbnrDR5gGDkg4763ZMtTHOAzSeD7si1d
  SEARCH_ENDPOINT=https://user24-ai-search-001.search.windows.net
  SEARCH_INDEX_NAME=rag-cij-005
  
  AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=user24storageaccount001;AccountKey=DKvTy0RxJ3S0rvpfsCsRG9ZLTdRuKd5K6B1GBKXrH4hrYb2vRqNTeXbu5U6Un9/sVPV+QjeGyM0X+AStBf+XYA==;EndpointSuffix=core.windows.net"
  AZURE_BLOB_CONTAINER=ia-documetns
  ```

- streamlit.sh
  ```
  추가 필요해보임 : matplotlib
  pip install streamlit openpyxl azure-identity pandas python-dotenv azure-storage-blob azure-search-documents openai python-docx matplotlib
  python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
  
  혹은
  python -m pip install -r requirements.txt
  python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
  ```
  
- 기타

  > 실습 수업자료 : https://github.com/KoreaEva/KTds2 



----


### 📌 프로젝트 개요

#### 💡 문제 정의

- 다수의 고객 요구사항이 매달 발생하며, 파트별로 중복/누락/영향 분석이 수작업으로 진행됨.

  - (동일한 요구사항에서 파생된 개발/운영이어도 각 파트에서는 본인파트 외에 다른 파트 영향도를 모르는 경우가 존재)

  - A파트 개발 완료되어 배포하려고 하는데, B파트 영향도 파악이 안되서 배포 밀리는 현상

- 산출물은 문서 형태로 존재하지만 재사용이 어렵고 검색이 비효율적임.

  - (내 과제지만 다른 파트 영향도를 알고 싶고, 관련된 내용도 알고 싶은데 직접 물어봐야하고 알기 어려움.)



#### 👤 대상 사용자

- 사업부서 및 개발/운영 담당자



#### 🧩 솔루션 개요

- **Azure AI Search + OpenAI + Streamlit**을 기반으로
   ▶ 유사 요구사항 자동 검색
   ▶ 업무 영향 파트 검색
- 벡터 검색 기반 RAG(검색 증강 생성) 구조 적용



----


### 🏗 시스템 아키텍처

#### 🧩 사전작업 및 프로세스 :

- Azure Storage : 정제된 문서 업로드
  - 파트별로 규격이 달라서 일관된 규격 필요
- Cognitive Search + OpenAI : 문서의 인덱싱 및 벡터 임베딩
- GPT : 파트 연관도 및 요약 응답 생성
- Streamlit : 프론트엔드 시각화
- Azure Web App : 배포




----


### 🚀 핵심 기술 포인트

#### 🎯 1. 자연어 임베딩 기반 유사 문서 검색

- `text-embedding-3-small` 모델을 Azure OpenAI에 배포하고 AI Search에 연결
- 요구사항 → 벡터 임베딩 생성 → 인덱스 탐색 → 관련 문서 추출(top-5)

#### 🧠 2. 문서 요약 및 파트 영향도 자동 추론

- 기존 문서 → GPT 기반 요약 
- GPT에 파트별 정의 프롬프트 + 요구사항을 전달해 `연관도 / 영향도` 수치화
- 결과를 시각화하고 업무 배분/장애 대응에 활용



----


### 💻 라이브 데모 화면

> 웹앱 : https://user24-webapp-001-emafesfpf9e9ehae.eastus-01.azurewebsites.net/

> (최종) 웹앱2 : https://user24-webapp-cij-002-cwhmdxf6atfsb9e4.eastus-01.azurewebsites.net/

> (최최종) 웹앱3 : https://user24-webapp-003-d7a2ggejhkhdbwgn.eastus-01.azurewebsites.net/



#### 🧩 진행 중 이슈사항

1. 일반 문서 업로드 시 전체 내용이 context로 들어가서 파싱이 안되는 이슈 -> 검색이 하나도 되지 않았음.

​	=> 일관된 규격으로 데이터 전처리 후 재업로드

- 전처리 후 규격

```json
{
  "요구사항 ID": "REQ001",
  "요구사항 제목": "Eius ipsum a illo.",
  "요구사항 요약": "백오피스 내 물류 LAN 기능을 개선하기 위한 화면 개발 요청",
  "요청자": "(유) 김",
  "주요 기능": "기능 설정, 관리, 로그 기록",
  "조건": "관리자 접근 제한, 로그 이력, 롤백 가능",
  "UI/UX 요구사항": "단일 화면 내 상태 확인, 오류 시 시각적 안내",
  "개발 요건": [
    "상세 기능 정의서 기반 화면 설계",
    "백엔드 로직 처리 및 DB 테이블 설계",
    "UI/UX 설계 반영"
  ],
  "개발 결과": "기능이 개발 완료되어 테스트 검증 후 운영 반영됨"
}
```

- **추가) IA문서 업로드 시 정제된 규격으로 변경해주기**



2. 프롬프트 개선

| AS-IS                                                        |
| ------------------------------------------------------------ |
| [시스템 역할 지시] 너는 IT 회사의 소프트웨어 운영팀에서 사용하는 AI 분석 도우미야. |

```python
[시스템 역할 지시]
너는 소프트웨어 운영팀의 AI 분석 도우미야.
고객 요구사항을 기반으로 어떤 파트(업무 영역)와 얼마나 연관 있는지를 판단해줘.

[입력 요구사항]
요구사항: "모니터링 시스템에서 오늘 기록만 빨간색으로 표기되도록 변경해줘."

[유사 문서 예시]

1. 요구사항: "장애 대응 모니터링에 시간 조건을 설정할 수 있도록 개발"
   연관 파트: 모니터링, 장애대응
2. 요구사항: "요금조회 화면에서 금일 사용량을 빨간색으로 표시"
   연관 파트: 요금정보, UI개발

[출력 형식]
{
 "모니터링": 0.8,
 "화면개발": 0.6,
 "요금책정": 0.2,
 "수납": 0.0,
 "장애대응": 0.4
}
```

| TO-BE |
| ----- |

```python
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
```

- **추가) 최적의 프롬프트 찾기**



#### 🧩 데이터 샘플

- 샘플 파트 목록

> parts = ["요금책정", "수납", "화면개발", "모니터링", "운영"]

- 샘플 고객 요구사항

> "고객의 다양한 요구에 대응하기 위해 시스템 개선이 필요합니다.",
>"업무 효율성 향상을 위해 해당 기능의 고도화가 요청되었습니다.",
> "장애 대응 시간 단축 및 신뢰성 확보를 위해 본 요구사항이 제출되었습니다.",
> "법적 컴플라이언스를 위해 새로운 기능이 반영되어야 합니다.",
> "사용자 경험(UX) 개선 요청이 지속적으로 제기되어 기능 개선이 필요합니다."

- 요구사항 입력 예시
  - 신규 요금제 개발하고 싶어
  - 모니터링 시스템에서 오늘 데이터는 빨간색으로 표기하고 싶어
  - 장애 대응 시스템에서 통계 기능을 추가하고 싶어



----


### 📈 향후 개선 및 확장 계획

#### 🔧 기능 확장

- 파트 간 `업무 중복/누락` 자동 탐지
- **추가) 비즈니스 로직 추가 → 상세 개발 내용 추천**
- 개발 완료 문서에 대한 GPT 기반 `QA 자동화`

#### 🧠 모델 활용 고도화

- `custom prompt tuning` → GPT의 파트별 분석 성능 개선
- 사용자 피드백 반영한 RAG 최적화



----

