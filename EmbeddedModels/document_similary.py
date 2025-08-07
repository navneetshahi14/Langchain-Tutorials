from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimension=300)

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "Rohit Sharma is an Indian cricketer known for his effortless stroke play and record-breaking double centuries.",
    "MS Dhoni is an Indian cricketer known for his calm captaincy and exceptional finishing skills.",
    "Sachin Tendulkar is an Indian cricketer known for being the highest run-scorer in international cricket history.",
    "Jasprit Bumrah is an Indian cricketer known for his deadly yorkers and world-class fast bowling.",
    "Hardik Pandya is an Indian cricketer known for his explosive all-round performances and power-hitting.",
    "KL Rahul is an Indian cricketer known for his versatile batting across all formats."
]

query = "tell me about virat kohli"

doc_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0]

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(document[index])
print("similarity score is: ",score)