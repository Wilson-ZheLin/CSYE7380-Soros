import os
import torch
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

class RAGSystem:
    def __init__(self, embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", persist_directory="chroma_db"):
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.embeddings = TransformersEmbeddings(model_name=embedding_model_name)

        # Load existing database if it exists
        if os.path.exists(persist_directory):
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            self.vectordb = None

    def load_csv(self, csv_path, question_col=0, answer_col=1, chunk_size=1000, chunk_overlap=100):

        # Read CSV file
        df = pd.read_csv(csv_path, header=None)

        # Prepare document list
        documents = []
        for idx, row in df.iterrows():
            question = row[question_col]
            answer = row[answer_col]
            # Use question as metadata, answer as content
            doc = Document(page_content=answer, metadata={"question": question, "source": csv_path})
            documents.append(doc)

        # Split texts
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = text_splitter.split_documents(documents)

        # Create vector database
        self.vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"Index created with {len(docs)} document chunks")

    def search(self, query, k=1):

        if self.vectordb is None:
            raise ValueError("Please load a CSV file or initialize vector database first")

        results = self.vectordb.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "question": doc.metadata.get("question", ""),
                "source": doc.metadata.get("source", ""),
                "score": score
            })

        return formatted_results

class TransformersEmbeddings(Embeddings):
    
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]

        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy()

    def embed_documents(self, texts):
        embeddings = self.encode_text(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.encode_text(text)[0]
        return embedding.tolist()

if __name__ == "__main__":
    # Please set the path to your CSV file
    file_path = "/Users/zhe/Downloads/Soros-QNA-Pairs.csv"

    rag_system = RAGSystem()
    rag_system.load_csv(file_path)
    results = rag_system.search("Who is George Soros?")
    for i, result in enumerate(results):
        print("-" * 50)
        print(f"Result {i+1}:")
        print(f"Content: {result['content']}")
        print(f"Original question: {result['question']}")
        print(f"Relevance score: {result['score']}")
        print("-" * 50, end="\n\n")