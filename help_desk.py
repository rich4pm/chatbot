import sys
import load_db
import collections
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np

class HelpDesk:
    """Create the necessary objects to create a QARetrieval chain"""

    def __init__(self, new_db=True):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()

        if self.new_db:
            print("Setting up a new database...")
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            print("Loading existing database...")
            self.db = load_db.DataLoader().get_db(self.embeddings)

        self.retriever = self.db.as_retriever()
        self.retrieval_qa_chain = self.get_retrieval_qa()

        # Initialize the SentenceTransformer model for relevance checking
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_template(self):
        template = """
        Given this text extracts:
        -----
        {context}
        -----
        Please answer with the following question:
        Question: {question}
        Helpful Answer:
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.template, input_variables=["context", "question"]
        )
        return prompt

    def get_embeddings(self) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings()
        return embeddings

    def get_llm(self):
        llm = OpenAI()
        return llm

    def get_retrieval_qa(self):
        chain_type_kwargs = {"prompt": self.prompt}
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        return qa

    def retrieval_qa_inference(self, question, verbose=True):
        query = {"query": question}
        print(f"Querying with question: {question}")
        answer = self.retrieval_qa_chain(query)

        # Evaluate relevance of the answer
        if not self.is_relevant_answer(question, answer["source_documents"]):
            return "Sorry, I couldn't find any relevant information to answer your question.", []

        sources = self.list_top_k_sources(answer, k=2)

        if verbose:
            print(f"Sources: {sources}")

        return answer["result"], sources

    def is_relevant_answer(self, query, source_documents, threshold=0.4):
        print(f"Evaluating relevance of the documents...")
        query_embedding = self.similarity_model.encode(query, convert_to_tensor=True)
        doc_embeddings = [self.similarity_model.encode(doc.page_content, convert_to_tensor=True) for doc in source_documents]
        
        # Compute cosine similarities
        similarities = [util.pytorch_cos_sim(query_embedding, doc_embedding).item() for doc_embedding in doc_embeddings]
        
        print(f"Similarities: {similarities}")
        # Check if any similarity score meets the threshold
        return any(score >= threshold for score in similarities)

    def list_top_k_sources(self, answer, k=2):
        sources = [
            f'[{res.metadata["title"]}]({res.metadata["source"]})'
            for res in answer["source_documents"]
        ]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

            if len(distinct_sources) == 1:
                return f"Here is the source that might be useful to you: \n- {distinct_sources_str}"

            elif len(distinct_sources) > 1:
                return f"Here are {len(distinct_sources)} sources that might be useful to you: \n- {distinct_sources_str}"

        return "Sorry, I couldn't find any resources to answer your question."
