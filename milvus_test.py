# ref1: https://milvus.io/docs/build-rag-with-milvus.md
# ref2: https://devocean.sk.com/blog/techBoardDetail.do?ID=165368

#%%
import os
import json
import urllib.request
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

print(openai_api_key)  # 테스트 출력
#%%
# prepare the data
url = "https://raw.githubusercontent.com/milvus-io/milvus/master/DEVELOPMENT.md"
file_path = "./Milvus_DEVELOPMENT.md"

if not os.path.exists(file_path):
    # download the file
    urllib.request.urlretrieve(url, file_path)
    
with open(file_path, "r") as file:
    # read the file
    file_text = file.read()

# split the text into lines
text_lines = file_text.split("# ")

# %%
from openai import OpenAI

openai_client = OpenAI()

# %%
# define a function to embed text
def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

# %%
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

# %%
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

connections.connect("default", host="localhost", port="19530")

collection_name = "openai_embeddings"

# 기존 컬렉션이 있다면 삭제하는 코드
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Existing collection {collection_name} dropped.")

if not utility.has_collection(collection_name):
    ### 사전 필드 스키마 정의 ###
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary = True, auto_id = True),  
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    #컬렉션 스키마 정의
    schema = CollectionSchema(fields, "openai embeddings collection")
    
    # fields = [
    #     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    # ]
    # # 동적 필드 스키마 정의 
    # schema = CollectionSchema(
    #     fields=[],
    #     description="openai embeddings collection with dynamic schema",
    #     auto_id=False,  # 자동 ID 생성 비활성화
    #     enable_dynamic_field=True  # 동적 필드 활성화
    # )
    
    # 컬렉션 생성
    mv_test = Collection(
        collection_name,
        schema, 
        # metric_type="IP",
        # consistency_level="Strong"
        )
    print(f"Collection {collection_name} created successfully")
   
else:
    mv_test = Collection(collection_name)
    print(f"Collection {collection_name} already exists")
 # 컬렉션 스키마 출력
print("Collection schema:")
for field in mv_test.schema.fields:
    print(f"Field name: {field.name}, Data type: {field.dtype}, Is primary: {field.is_primary}, Auto ID: {field.auto_id}")
#%%
# 클라이언트 연결 확인
if connections.has_connection("default"):
    print("Connected to Milvus server successfully.")
else:
    print("Failed to connect to Milvus server.")

# check if the collection is created
print(f"Does collection {collection_name} exist in Milvus: {collection_name in utility.list_collections()}")
# %%
# Insert the embeddings
from tqdm import tqdm
import pandas as pd

'''
주의) 
1. 데이터를 삽입할 때, 데이터를 리스트 형태로 변환하여 삽입해야 함
2. 컬렉션 생성 시 자동 생성된 id 필드를 포함하는 필드 정의가 불필요할 수 있음. 확인 할 것.
* schema 필드 정의 시 auto_id를 애초에 끄는 것고 괜찮을 듯
'''
# Insert the embeddings
# data_ids = []
data_vectors = []
data_texts = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    #data_ids.append(i)
    data_vectors.append(emb_text(line))
    data_texts.append(line)

# 데이터를 리스트 형태로 변환
entities = [
    # data_ids,
    data_vectors,
    data_texts
]

# 각 값의 타입 확인
for idx, entity in enumerate(entities):
    print(f"Type of entities[{idx}]: {type(entity)}")
    if len(entity) > 0:
        print(f"Type of entities[{idx}][0]: {type(entity[0])}")
        
# 데이터를 컬렉션에 삽입
insert_result = mv_test.insert(entities)
mv_test.flush()

print(f"Inserted {len(insert_result.primary_keys)} records into the collection '{collection_name}'")
# %%
# 입력한 결과 조회
# %%
# Index 생성
# define fmt for better print
fmt = "======================== {} ========================\n"
print(fmt.format("Start creating index"))

index = {
    "index_type": "IVF_FLAT", # VF_FLAT divides the vector data into nlist cluster units
    "metric_type": "L2", # Euclidean distance
    "params": {"nlist": 128} # number of cluster unit (1~65536, default=128)
}

mv_test.create_index("vector", index)
# %%
# Index Loading & Similarity Search
mv_test.load()

question = "How to build a recommendation system?" # 검색 질문

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
} # nprobe: number of clusters to search, default=10

search_result = mv_test.search(
                        data = [
                            emb_text(question)
                        ], # 검색할 데이터, list로 넣어줘야함
                        anns_field= "vector",
                        param = search_params, 
                        limit=3,
                        output_fields=["text"],
                        expr=None)

'''
TypeError: 'Hit' object is not subscriptable 오류는 
Milvus의 Hit 객체가 리스트나 딕셔너리처럼 인덱싱이 불가능하기 때문에 발생. 
대신 Hit 객체의 속성을 사용하여 데이터를 접근해야 함
'''
# search_result에서 각 Hit 객체의 entity와 distance 속성을 사용하여 검색 결과를 출력
retrieved_lines_with_distances = [
    (hit.entity.get("text"), hit.distance) for hit in search_result[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
# %%
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)
SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

# response
response = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
# %%
res = mv_test.describe_collection()

print(res)

# %%
