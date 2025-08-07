from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Lucknow is the capital of UttarPradesh",
    "France is capital of Paris"
]

result = embedding.embed_query(documents)

print(str(result))

