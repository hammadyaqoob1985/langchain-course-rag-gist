import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core import embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("Ingesting...")
    loader = TextLoader(
        "C:/Users/hamma/cursorProjects/langchain-course-rag-gist/mediumblog1.txt",
        encoding='UTF-8')
    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )

    print("finish")

if __name__ == "__main__":
    main()
