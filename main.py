from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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
    
    format_instructions = {
        "requirement": "The requirement that was extracted from the text.",
        "assumption": "The assumption that was made to extract the requirement."
    }
    
    template = """
    You are a helpful assistant. You will be given a chunk of text that is part of an RFP document. Here is a summary of the RFP: {summary}. Extract the cloud related requirements from the text. The requirements will be ambiguous and you will need to make assumptions. 
    You should return a JSON object with the following fields:
    - requirement: The requirement that was extracted from the text.
    - description: The description of the requirement that was extracted from the text.
    - assumption: The assumption that was made to extract the requirement.
    Provide the output in the following JSON format:
    {format_instructions}
    Example:
    RFP Text:
    "The system must be able to scale dynamically to meet changing demands. It is expected to handle a minimum of 10,000 concurrent users."
    Extracted Requirement:
    Answer:
    [
        {
            "requirement": "Dynamic scaling to handle 10,000 concurrent users.",
            "assumption": "We assume that 'concurrent users' refers to simultaneous active sessions, and that the system needs to maintain acceptable performance (response times under 2 seconds) under this load."
        },
        {
            "requirement": "The system must be able to scale dynamically to meet changing demands.",
            "assumption": "We assume that 'dynamic scaling' refers to the ability of the system to adjust its resources based on demand, such as adding or removing instances in response to changes in traffic or workload."
        }
    ]
    
    RFP Text:   
    {text}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"], partial_variables={"format_instructions": format_instructions, "summary": summary})
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": chunk})
    
    return response['content']
    

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


def helloword():
    print("Hello World")