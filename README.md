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
  ```
  
- 기타

  > 실습 수업자료 : https://github.com/KoreaEva/KTds2 

----



### 📌 Slide 1: 프로젝트 개요

#### 💡 문제 정의

- 다수의 고객 요구사항이 매달 발생하며, 파트별로 중복/누락/영향 분석이 수작업으로 진행됨.

- (동일한 요구사항에서 파생된 개발/운영이어도 각 파트에서는 본인파트 외에 다른 파트 영향도를 모르는 경우가 존재)

- 산출물은 문서 형태로 존재하지만 재사용이 어렵고 검색이 비효율적임.

- (내 과제지만 다른 파트 영향도를 알고 싶고, 관련된 내용도 알고 싶음)

  

#### 👤 대상 사용자

- 소프트웨어 개발/운영 부서의 파트 리더, 분석가, 담당자
- 요금제, 수납, 화면개발, 모니터링, 운영 파트를 담당하는 실무자



#### 🧩 솔루션 개요

- **Azure AI Search + OpenAI + Streamlit**을 기반으로
   ▶ 유사 요구사항 자동 검색
   ▶ 업무 영향 파트 검색
- 벡터 검색 기반 RAG(검색 증강 생성) 구조 적용



----



### 🏗 Slide 2: 시스템 아키텍처

#### 🧩 연동 구성요소:

- 정제된 문서 업로드 → Azure Storage
- 인덱싱 및 벡터 임베딩 → Cognitive Search + OpenAI
- 실시간 검색/요약 → Azure Function + GPT
- 프론트엔드 시각화 → Streamlit

![diagram_1](https://github.com/user-attachments/assets/6feb4793-6348-4c3b-a3e4-a9685d8a3872)


----



### 🚀 Slide 3: 핵심 기술 포인트

#### 🎯 1. 자연어 임베딩 기반 유사 문서 검색

- `text-embedding-3-small` 모델을 Azure OpenAI에 배포하고 Cognitive Search에 연결
- 요구사항 → 벡터 → 인덱스 탐색 → 관련 문서 추출

#### 🧠 2. 파트 영향도 자동 추론

- GPT에 파트별 정의 프롬프트 + 요구사항을 전달해 `연관도 / 영향도` 수치화
- 결과를 시각화하고 업무 배분/장애 대응에 활용

#### 🧱 3. 실시간 문서 요약 및 제목 자동 생성

- 기존 문서 → GPT 기반 요약 → `요구사항 제목` 자동 생성
- 유사 요구사항 검색 정확도 향상



----



### 💻 Slide 4: 라이브 데모 화면

> 웹앱 : https://user24-webapp-001-emafesfpf9e9ehae.eastus-01.azurewebsites.net/
> 웹앱2 : https://user24-webapp-cij-002-cwhmdxf6atfsb9e4.eastus-01.azurewebsites.net/



#### 🧩 진행 중 이슈사항

1. 일반 문서 업로드 시 전체 내용이 context로 들어가서 파싱이 안되는 이슈 -> 검색이 하나도 되지 않았음.

​	=> 데이터 전처리 후 재업로드

2. 프롬프트 개선

| AS-IS                                                        |
| ------------------------------------------------------------ |
| [시스템 역할 지시] 너는 IT 회사의 소프트웨어 운영팀에서 사용하는 AI 분석 도우미야. |

```
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

| TO-BE                                                        |
| ------------------------------------------------------------ |
| [시스템 역할 지시]너는 IT 회사의 소프트웨어 운영팀에서 사용하는 AI 분석 도우미야.  <br/>너의 역할은 다음과 같아:<br/>1. 고객이 입력한 요구사항과 유사한 산출물(IA 문서)을 보여주고,<br/>2. 그 문서들과 비교해봤을 때 어떤 파트(업무 영역)가 이 요구사항과 연관되어 있는지를 추론해,<br/>3. 각 파트별 연관도를 수치(0~1.0)로 출력해줘. (1.0에 가까울수록 연관이 높음)<br/><br/>[입력된 고객 요구사항]<br/>요구사항: "{{user_input}}"  ← 사용자의 실제 입력이 들어갈 자리<br/><br/>[IA 문서 예시]  <br/>※ 다음은 벡터 유사도로 검색된 실제 IA 산출물들이다. 각 문서에는 관련된 요구사항과 담당 파트가 포함되어 있다.<br/>{{retrieved_ia_docs}}<br/><br/>[분석 방식]<br/>- 각 IA 문서의 내용, 연관 파트를 참고하여 현재 요구사항이 어떤 파트와 관련이 있는지 판단해.<br/>- 기능 영역, UI/UX, 모니터링, 운영 자동화, 수납, 요금 등의 키워드를 활용해서 판단하되, 단순 키워드 일치보다 문맥의 목적과 작업 흐름을 중점으로 판단해.<br/>- 관련된 파트는 여러 개일 수 있으며, 각 파트에 대해 0.0 ~ 1.0 사이 연관도 점수를 정수 두 자릿수 소수로 표시해줘.<br/>- 반드시 실제 산출물 문서 내용과 비교하며 판단해.<br/><br/>[출력 형식 예시]<br/>요구사항 분석 결과:<br/>- 요금정보 파트: 0.82<br/>- 화면개발 파트: 0.65<br/>- 모니터링 파트: 0.91 ← 가장 연관도 높음<br/><br/>[제한 사항]<br/>- 반드시 정해진 파트 목록 내에서만 판단해줘: ["요금정보", "수납", "화면개발", "모니터링", "운영"]<br/>- 문서가 없거나 문맥이 애매할 경우, 해당 파트는 생략해도 돼. |



#### 🧩 데이터 샘플

- 샘플 파트 목록

> parts = ["요금책정", "수납", "화면개발", "모니터링", "운영"]

- 고객 요구사항

    "고객의 다양한 요구에 대응하기 위해 시스템 개선이 필요합니다.",
    "업무 효율성 향상을 위해 해당 기능의 고도화가 요청되었습니다.",
    "장애 대응 시간 단축 및 신뢰성 확보를 위해 본 요구사항이 제출되었습니다.",
    "법적 컴플라이언스를 위해 새로운 기능이 반영되어야 합니다.",
    "사용자 경험(UX) 개선 요청이 지속적으로 제기되어 기능 개선이 필요합니다."

- 정제된 데이터

    "id": f"REQ{str(i).zfill(3)}",
    "part": part,
    "요구사항": f"[{part}] {random.choice(sample_needs)}",
    "요구사항 배경 및 상세 내용": random.choice(sample_backgrounds),
    "개발 요건": random.choice(sample_requirements),
    "개발 결과": random.choice(sample_results)

- 프롬프트

>[시스템 역할 지시]
>너는 소프트웨어 운영팀의 AI 분석 도우미야.
>고객 요구사항을 기반으로 어떤 파트(업무 영역)와 얼마나 연관 있는지를 판단해줘.
>
>[입력 요구사항]
>요구사항: "모니터링 시스템에서 오늘 기록만 빨간색으로 표기되도록 변경해줘."
>
>[유사 문서 예시]
>
>1. 요구사항: "장애 대응 모니터링에 시간 조건을 설정할 수 있도록 개발"
>     연관 파트: 모니터링, 장애대응
>2. 요구사항: "요금조회 화면에서 금일 사용량을 빨간색으로 표시"
>     연관 파트: 요금정보, UI개발
>
>[출력 형식]
>{
>  "모니터링": 0.8,
>  "화면개발": 0.6,
>  "요금책정": 0.2,
>  "수납": 0.0,
>  "장애대응": 0.4
>}



- 요구사항 입력 예시
  - 신규 요금제 개발하고 싶어
  - 모니터링 시스템에서 오늘 데이터는 빨간색으로 표기하고 싶어
  - 장애 대응 시스템에서 통계 기능을 추가하고 싶어

----



### 📈 Slide 5: 향후 개선 및 확장 계획

#### 🔧 기능 확장

- 파트 간 `업무 중복/누락` 자동 탐지
- 개발 완료 문서에 대한 GPT 기반 `QA 자동화`

#### 🧠 모델 활용 고도화

- `custom prompt tuning` → GPT의 파트별 분석 성능 개선
- 사용자 피드백 반영한 RAG 최적화



----

