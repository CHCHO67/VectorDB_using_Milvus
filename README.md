# VectorDB_using_Milvus
## Milvus와 OpenAI를 이용한 문서 임베딩 및 검색 시스템

이 프로젝트는 Milvus와 OpenAI API를 사용하여 문서 임베딩 및 검색 시스템을 구현하는 예제입니다. 문서를 파싱하고, 임베딩 벡터를 생성하여 Milvus에 저장한 후, 사용자의 질문에 대해 관련 문서를 검색하고 응답을 생성하는 과정을 포함합니다.

### 주요 기능

1. **환경 변수 설정**: `.env` 파일에서 OpenAI API 키를 로드합니다.
2. **Milvus 서버 연결**: Milvus 서버에 연결하고, 컬렉션을 생성하거나 기존 컬렉션에 연결합니다.
3. **문서 임베딩**: 문서 파일 또는 폴더를 입력받아 텍스트를 파싱하고 임베딩 벡터를 생성합니다.
4. **데이터 삽입**: 생성된 임베딩 벡터를 Milvus 컬렉션에 삽입합니다.
5. **인덱스 생성**: Milvus 컬렉션에 인덱스를 생성하여 검색 성능을 향상시킵니다.
6. **문서 검색**: 사용자의 질문에 대해 Milvus에서 관련 문서를 검색합니다.
7. **OpenAI GPT-3.5를 이용한 응답 생성**: 검색된 문서 내용을 기반으로 사용자의 질문에 대한 응답을 생성합니다.

### 설치 및 실행 방법

1. **필수 라이브러리 설치**:
    ```bash
    pip install pymilvus openai python-dotenv tqdm
    ```

2. **프로젝트 디렉토리 구조**:
    ```
    project_root/
    ├── milvus_openai_module.py
    ├── .env
    └── dataset/
        ├── file1.md
        ├── file2.md
        └── ...
    ```

3. **.env 파일 설정**:
    프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 OpenAI API 키를 설정합니다.
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **스크립트 실행**:
    ```bash
    python milvus_openai_module.py
    ```

### 코드 설명

#### `Milvus_Embedding` 클래스

- **초기화**: OpenAI API 키와 Milvus 서버 연결 설정.
- **Milvus 서버 연결**: Milvus 서버에 연결합니다.
- **데이터 준비**: 파일이나 폴더에서 텍스트를 읽어와 파싱합니다.
- **컬렉션 생성**: Milvus 컬렉션을 생성합니다.
- **텍스트 임베딩**: OpenAI API를 사용하여 텍스트를 임베딩 벡터로 변환합니다.
- **데이터 삽입**: 임베딩 벡터를 Milvus 컬렉션에 삽입합니다.
- **인덱스 생성**: Milvus 컬렉션에 인덱스를 생성합니다.

#### `Milvus_Retrieving` 클래스

- **초기화**: OpenAI API 키와 Milvus 서버 연결 설정.
- **Milvus 서버 연결**: Milvus 서버에 연결합니다.
- **컬렉션 연결**: 기존 Milvus 컬렉션에 연결합니다.
- **텍스트 임베딩**: OpenAI API를 사용하여 질문을 임베딩 벡터로 변환합니다.
- **데이터 검색**: Milvus에서 임베딩 벡터를 사용하여 관련 문서를 검색합니다.
- **응답 생성**: 검색된 문서 내용을 기반으로 OpenAI GPT-3.5를 이용하여 응답을 생성합니다.

### 주요 함수 설명

- `prepare_data(file_path)`: 파일이나 폴더에서 텍스트를 읽어와 파싱합니다.
- `_parse_file(file_path)`: 파일을 읽어와 텍스트를 파싱합니다.
- `create_collection(collection_name, embedding_dim)`: Milvus 컬렉션을 생성합니다.
- `embed_text(text)`: 텍스트를 임베딩 벡터로 변환합니다.
- `insert_data(text_lines)`: 임베딩 벡터를 Milvus 컬렉션에 삽입합니다.
- `create_index()`: Milvus 컬렉션에 인덱스를 생성합니다.
- `search_data(query, limit=3)`: Milvus에서 관련 문서를 검색합니다.
- `chat_with_context(context, question, system_prompt, user_prompt)`: 검색된 문서 내용을 기반으로 OpenAI GPT-3.5를 이용하여 응답을 생성합니다.

### 예제 실행

1. `.env` 파일에서 OpenAI API 키를 로드합니다.
2. Milvus 서버에 연결합니다.
3. 파일이나 폴더를 입력받아 문서를 파싱하고 임베딩 벡터를 생성합니다.
4. Milvus 컬렉션을 생성하고 데이터를 삽입합니다.
5. Milvus 컬렉션에 인덱스를 생성합니다.
6. 사용자의 질문에 대해 관련 문서를 검색하고 응답을 생성합니다.

### 참고 자료

- [Milvus Documentation](https://milvus.io/docs/)
- [OpenAI API Documentation](https://beta.openai.com/docs/)

이 프로젝트를 통해 Milvus와 OpenAI API를 사용한 문서 임베딩 및 검색 시스템을 구현할 수 있습니다. 다양한 문서 형식에 대해 파싱 로직을 추가하여 확장할 수 있으며, OpenAI의 다양한 모델을 사용하여 응답 생성의 품질을 향상시킬 수 있습니다.