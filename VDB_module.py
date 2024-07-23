# milvus_openai_module.py

import os
import re
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from openai import OpenAI

'''
Embedding
'''
class Milvus_Embedding:
    def __init__(self, openai_api_key, collection_name, milvus_host='localhost', milvus_port='19530'):
        self.collection_name = collection_name
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.connect_milvus(milvus_host, milvus_port)

    def connect_milvus(self, host, port):
        connections.connect("default", host=host, port=port)
        if connections.has_connection("default"):
            print("Connected to Milvus server successfully.")
        else:
            raise ConnectionError("Failed to connect to Milvus server.")

    def prepare_data(self, file_path):
        if os.path.isdir(file_path):
            all_text_lines = []
            for filename in os.listdir(file_path):
                full_path = os.path.join(file_path, filename)
                if os.path.isfile(full_path):
                    text_lines = self._parse_file(full_path)
                    all_text_lines.extend(text_lines)
            return all_text_lines
        elif os.path.isfile(file_path):
            return self._parse_file(file_path)
        else:
            raise ValueError(f"{file_path} is not a valid file or directory")

    def _parse_file(self, file_path):
        with open(file_path, "r") as file:
            file_text = file.read()
        # 정규 표현식을 사용하여 #, ##, ### 헤더로 텍스트를 분할
        pattern = re.compile(r'(#{1,3} .+?)(?=\n#{1,3} |\Z)', re.DOTALL)
        sections = pattern.findall(file_text)
        return sections
    
    def create_collection(self, collection_name, embedding_dim):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Existing collection {collection_name} dropped.")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary = True, auto_id = False),  
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, "openai embeddings collection")
        self.collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")
        print(f"Collection {collection_name} created successfully")
        print("Collection schema:")
        for field in self.collection.schema.fields:
            print(f"Field name: {field.name}, Data type: {field.dtype}, Is primary: {field.is_primary}, Auto ID: {field.auto_id}")

    def embed_text(self, text):
        return self.openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

    def insert_data(self, text_lines):
        data_ids = []
        data_vectors = []
        data_texts = []
        for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
            data_ids.append(i)
            data_vectors.append(self.embed_text(line))
            data_texts.append(line)
        entities = [data_ids, data_vectors, data_texts]
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        print(f"Inserted {len(insert_result.primary_keys)} records into the collection '{self.collection.name}'")

    def create_index(self):
        fmt = "======================== {} ========================\n"
        print(fmt.format("Start creating index"))
        index = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        self.collection.create_index("vector", index)


class Milvus_Retrieving:
    def __init__(self, openai_api_key, collection_name, milvus_host='localhost', milvus_port='19530'):
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        #openai.api_key = self.openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.connect_milvus(milvus_host, milvus_port)
        self.connect_collection()

        
    def connect_milvus(self, host, port):
        connections.connect("default", host=host, port=port)
        if connections.has_connection("default"):
            print("Connected to Milvus server successfully.")
        else:
            raise ConnectionError("Failed to connect to Milvus server.")

    def connect_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"Connected to existing collection {self.collection_name}")
        else:
            raise ValueError(f"Collection {self.collection_name} does not exist")

    def embed_text(self, text):
        return self.openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    
    def search_data(self, query, limit=3):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        query_embedding = self.embed_text(query)
        search_result = self.collection.search(data=[query_embedding], anns_field="vector", param=search_params, limit=limit, output_fields=["text"])
        return [(hit.entity.get("text"), hit.distance) for hit in search_result[0]]

    def chat_with_context(self, context, question, system_prompt, user_prompt):
        USER_PROMPT = user_prompt.format(context=context, question=question)
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
        )
        return response.choices[0].message.content
    
def main():
    '''
    OPENAI_API 키를 환경 변수에서 가져와서 사용
    1. .env 파일을 찾고, 없으면 예외 발생
    2. .env 파일이 있으면 .env 파일에 설정된 OPENAI_API_KEY 사용(환경 변수를 덮어 쓰도록)
    3. OPENAI_API_KEY가 없으면 예외 발생
    '''
    env_path = find_dotenv()
    print(env_path)
    if not env_path:
        raise FileNotFoundError(".env file not found")
    
    # override가 없으면 환경 변수를 우선한다.
    load_dotenv(env_path, override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    print(openai_api_key)
    
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    '''
    Milvus를 이용한 문서 임베딩 과정
        1. Embedding 객체를 생성하고, Milvus 서버에 연결
        2. 파일 입력
        3. 문서 파싱
        4. 컬렉션 생성
        5. 데이터 삽입
        6. 인덱스 생성
    '''
    collection_name = "mondrian_hrm"
    embedding = Milvus_Embedding(openai_api_key, collection_name)
    
    # 파일 입력
    file_path = "/mnt/sdb/X2_MonGPT_VectorDB/dataset"
    # 문서 파싱부, 현재는 MD 파일 파싱, 
    # 차후 문서 포맷에 특화된 파싱 부로 대체 할 수 있음
    text_lines = embedding.prepare_data(file_path)
    
    test_embedding = embedding.embed_text("This is a test")
    embedding_dim = len(test_embedding)
    
    embedding.create_collection(collection_name, embedding_dim)
    embedding.insert_data(text_lines)
    embedding.create_index()
    embedding.collection.load()
    
    '''
    Milvus를 이용한 문서 검색 과정
        1. Retrieving 객체를 생성하고, Milvus 서버에 연결
        2. 질문 입력
        3. 데이터 검색
        4. context 추출
        5. prompt 설정
        6. Response 생성
    '''
    retrieving = Milvus_Retrieving(collection_name=collection_name, openai_api_key=openai_api_key)
    # 질문을 받을 수 있도록 변경 필요
    test_question = "경조사 규정을 알려줘"
    
    # 검색 결과를 받아옴
    results = retrieving.search_data(test_question)
    
    # results 리스트에서 검색된 텍스트 부분만 추출하고, 이를 하나의 큰 문자열로 결합하여 context 변수에 저장.
    # 이 context 변수는 이후의 프롬프트에서 검색된 문맥 정보로 사용.
    context = "\n".join([res[0] for res in results])
    
    
    # 프롬프트 입력받을 수 있도록 변경 필요
    SYSTEM_PROMPT = "Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided."
    USER_PROMPT = "Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.\n<context>\n{context}\n</context>\n<question>\n{question}\n</question>"
    
    # chat_with_context 함수를 사용하여 OpenAI GPT-3.5를 이용한 대화 생성
    response = retrieving.chat_with_context(context, test_question, SYSTEM_PROMPT, USER_PROMPT)
    
    print(response)

if __name__ == "__main__":
    main()

# %%
