from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
import time
import datetime
import math

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''

  # Round to the nearest second.
  elapsed_rounded = int(round(elapsed))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))

# Load file from a folder
dir_loader = DirectoryLoader('doc', glob="**/*.txt", show_progress=True, use_multithreading=True, loader_cls=TextLoader)
docs = dir_loader.load()
print(f"Number of documents: {len(docs)}")

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)
print(f"    Total chunks: {len(texts)}")

# embedding
t0 = time.time()
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

persist_directory = 'db-bge-py2'

## Here is the new embeddings being used
embedding = model_norm

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# Calculate elapsed time in minutes
elapsed_embed = format_time(time.time()-t0)

print("\nDone!")
print(f"    Embedding time: {elapsed_embed}")
print(f"    Embedding file is stored on {persist_directory}")