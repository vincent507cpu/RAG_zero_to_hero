import os
from pathlib import Path
# 第一步：消化数据
# 1. 读取数据
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

root = Path(__file__).parent.parent.absolute()
loader = TextLoader(os.path.join(root, 'data/paul_graham/paul_graham_essay.txt'))
data = loader.load()

# print(f'{len(data)} | {data[0].dict().keys()}')
# print(type(data[0]))
# print(data[0].dict().keys())
# print(len(data[0].page_content))

# 2. 转化
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(data)

# print(len(docs))
# print(docs[0])

# 3. 向量化
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
model_id = "BAAI/bge-small-en-v1.5"
embedding = HuggingFaceEmbeddings(model_name=model_id)

embed = embedding.embed_documents(docs[0].page_content)

# print(f'{len(embedding[0])}: {embedding[0][:5]} ...')

# 第二步：检索数据
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(docs, embed)
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

query = 'What did the author do growing up?'
retrieval = retriever.invoke(input=query)

# print(len(retrieval))

# for i, doc in enumerate(retrieval):
#     print(i, '\n', doc.page_content, '\n')

# 第三步：生成答案
from langchain_ollama.llms import OllamaLLM
llm = OllamaLLM(model="qwen2.5:1.5b")

context = '\n'.join([doc.page_content for doc in retrieval])

prompt = f'''
You are a helpful assistant.
Answer the question based on the following context:
{context}
Question: {query}
'''

answer = llm.invoke(prompt)
print(answer)