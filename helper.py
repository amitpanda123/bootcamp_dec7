__all__ = ['embedding_models_HF', 'get_API_embedding',  'fetch_arxiv_data', 'doc_embedding']
import requests
from bs4 import BeautifulSoup
from typing import Optional
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings




# For embeddings we use models on the MTEB leaderboard at https://huggingface.co/spaces/mteb/leaderboard


# Voyage, currently nr 1 (REQUIRES REGISTERING TO GET API KEY)
#!pip install -q voyageai
#import voyageai
#from langchain.embeddings import VoyageEmbeddings
#os.environ["VOYAGE_API_KEY"] = "..."
#voyageai.api_key = os.environ["VOYAGE_API_KEY"]


# Cohere, currently nr 2 (REQUIRES REGISTERING TO GET API KEY)
#import cohere
#Get your API key from www.cohere.com
#os.environ["COHERE_API_KEY"] = "..."


# Open source HuggingFace embeddings, below is currently nr 3 & 12 (NO REGISTRATION REQUIRED)
embedding_models_HF = [
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5"
    ]

def fetch_arxiv_data(url, subject):
    """
    Function for fetching articles from the arXiv. Expects a url pointing to the daily relese site of arXiv topics.
    Returns a list of dictionaries containing 'title', 'abstract' and 'arxiv_topic'.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print('Failed to retrieve data from', url)
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    papers = []

    cnt_found=0
    cnt_not_found=0
    for item in soup.find_all('div', class_='meta'):
      # We only extract info from articles with abstract in the listing. This includes cross-topic listings
      try:
        title = item.find('div', class_='list-title mathjax').text.replace('Title:', '').strip()
        abstract = item.find('p', class_='mathjax').text.strip()
        arxiv_topic = item.find('span', class_='primary-subject').text.strip()
        papers.append({'title': title, 'abstract': abstract, 'arxiv_topic': arxiv_topic, 'subject': subject})
        cnt_found+=1
      # We do not try to get abstract from replacements
      except:
        #print(f"NO ABSTRACT FOUND, DUE TO ARTICLE BEING A REPLACEMENT OF EARLIER SUBMISSION")
        cnt_not_found+=1

    print(f"Extracted abstract for {cnt_found} new articles from {subject}.\nThis excludes {cnt_not_found} replacements.")

    return papers


# Helper functions for embedding chunked text using various embedding models
#-------------------------------------------------------------------------------
def doc_embedding(
    embedding_model: str,
    model_kwargs: dict={'device': 'cpu'},
    encode_kwargs: dict={'normalize_embeddings': True},
    cache_folder: Optional[str]=None,
    multi_process: bool=False,
    ) -> HuggingFaceEmbeddings:
  """
  TBW...
  """
  embedder = HuggingFaceEmbeddings(
      model_name = embedding_model,
      model_kwargs = model_kwargs,
      encode_kwargs = encode_kwargs,
      cache_folder = cache_folder,
      multi_process = multi_process
  )
  return embedder


def get_API_embedding(text, model="text-embedding-ada-002"):
  """This function retrieves embedding vector from text string using various models"""
  text = text.replace("\n", " ")

  # OpenAI embeddings
  if model == "text-embedding-ada-002":
    client = OpenAI()
    embedding = client.embeddings.create(input = [text], model=model).data[0].embedding

  # Voyage embeddings
  elif model == 'voyage-01':
    voyage = VoyageEmbeddings(model=model, voyage_api_key=os.environ["VOYAGE_API_KEY"])
    embedding = voyage.embed_query(text)

  # Cohere embeddings
  elif model == "embed-english-v3.0":
    co = cohere.Client(os.environ["COHERE_API_KEY"])
    embedding = co.embed([text], input_type="search_document", model=model).embeddings

  elif model in embedding_models_HF:
    print("YES")
    embedder = doc_embedding(model)
    embedding = embedder.embed_query(text)

  else:
    embedding = [None]

  return embedding
