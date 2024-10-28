from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
# import pandas as pd
import uuid
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec  # Add ServerlessSpec to imports

load_dotenv()

# Function that summarizes the RFP to give Context to the LLM
def summarize_rfp(llm: ChatOpenAI, rfp: str) -> str:
    template="You are a helpful assistant. You will be given an RFP document. Summarize the RFP in a way that is easy for a language model to understand. The text will be given to LLMs as a context to analyze smaller chunks of the RFP more easily. RFP:\n{text}"
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": rfp})


# Function to split text into semantic chunks
def split_text_into_chunks(text: str) -> list[str]:
    """
    Split input text into semantic chunks using OpenAI embeddings.
    
    Args:
        text (str): The input text to be split into chunks
    Returns:
        list[str]: A list of text chunks split semantically
    """
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.split_text(text)
    return chunks

# Function that extracts cloud related requirements from the chunks
def extract_requirements_from_chunk(llm: ChatOpenAI, chunk: str, summary: str) -> dict:
    """
    Extract cloud-related requirements from a chunk of RFP text using an LLM.
    
    Args:
        llm (ChatOpenAI): The language model to use for extraction
        chunk (str): The text chunk to analyze
        summary (str): The summary of the RFP
    Returns:
        dict: A dictionary containing the extracted requirement and assumption
    """
    
    format_instructions = """{
        "requirement": "The requirement that was extracted from the text.",
        "assumption": "The assumption that was made to extract the requirement."
    }"""
    
    template = """
    You are a helpful assistant. You will be given a chunk of text that is part of an RFP document. Here is a summary of the RFP: {summary}. Extract the cloud related requirements from the text. The requirements will be ambiguous and you will need to make assumptions. 
    You should return a JSON object with the following fields:
    - requirement: The requirement that was extracted from the text.
    - description: The description of the requirement that was extracted from the text.
    - assumption: The assumption that was made to extract the requirement.
    Provide the output in the following JSON format:
    {format_instructions}

    RFP Text:   
    {text}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"], partial_variables={"format_instructions": format_instructions, "summary": summary})
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": chunk})
    

# Function that splits the RFP into chunks and extracts the requirements from each chunk
def extract_requirements_from_rfp(llm: ChatOpenAI, rfp: str) -> list[dict]:
    """
    Extract cloud-related requirements from an entire RFP document by splitting it into chunks
    and analyzing each chunk.
    
    Args:
        llm (ChatOpenAI): The language model to use for extraction
        rfp (str): The full RFP document text
        
    Returns:
        list[dict]: A list of dictionaries, each containing:
            - chunk: The original text chunk
            - requirements: The extracted requirements and assumptions for that chunk
    """
    summary = summarize_rfp(llm, rfp)
    chunks = split_text_into_chunks(rfp)
    requirements = [{"chunk": chunk, "requirements": extract_requirements_from_chunk(llm, chunk, summary)} for chunk in chunks]
    
    return requirements



# Initialize global instances
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
vector_store = PineconeVectorStore.from_existing_index(
    index_name="matrix-embeddings",
    embedding=embeddings
)

def evaluate_requirements_with_criteria(requirements: list[dict]) -> list[dict]:
    """
    Evaluates each requirement against criteria in the Pinecone database to determine relevance.
    Now accepts output from extract_requirements_from_rfp.
    """
    global llm, embeddings, vector_store
    relevant_criteria = []

    for requirement in requirements:
        # Iterate over the extracted requirements in each chunk
        for req in requirement['requirements']:
            # Perform a similarity search using the requirement text directly
            matches = vector_store.similarity_search(
                query=req['requirement'],  # Use the extracted requirement text as the query
                k=5  # Use k instead of top_k
            )
            
            for match in matches:
                # Retrieve necessary fields from the match metadata
                metadata = match.metadata  # Access the metadata from the Document object
                uuid = metadata['UUID']
                überbegriff = metadata['Überbegriff']
                thema = metadata['Thema']
                subthema = metadata['Subthema']
                text = match.page_content  # Use page_content for the text
                
                # Create a prompt for GPT-4o-mini
                prompt = f"""
                Prüfen Sie, ob das folgende Kriterium für die Beantwortung der Anforderung an einen Cloud-Service-Anbieter relevant ist:
                Anforderung: {req['requirement']}
                Kriterium:
                - Überbegriff: {überbegriff}
                - Thema: {thema}
                - Subthema: {subthema}
                - Beschreibung: {text}
                
                Antwort mit 'true' oder 'false'.
                """
                
                # Invoke the language model
                response = llm.invoke(prompt)
                is_relevant = response.content.strip().lower() == 'true'
                
                if is_relevant:
                    # Add relevant criteria to the list
                    relevant_criteria.append({
                        "UUID": uuid,
                        "Überbegriff": überbegriff,
                        "Thema": thema,
                        "Subthema": subthema,
                        "text": text,
                        "aws": metadata['AWS'],
                        "gcp": metadata['GCP'],
                        "azure": metadata['Azure']
                    })
    
    return relevant_criteria

# Example usage
if __name__ == "__main__":
    # Test requirement
    with open("RFP.txt", "r") as file:
        rfp = file.read()
    requirements = extract_requirements_from_rfp(llm, rfp)
    
    # Evaluate requirements with criteria
    relevant_criteria = evaluate_requirements_with_criteria(requirements)
    
    # Print or process the relevant criteria
    print(relevant_criteria)
