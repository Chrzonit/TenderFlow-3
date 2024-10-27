import pandas as pd
import uuid
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec  # Add ServerlessSpec to imports

# Lade Umgebungsvariablen
load_dotenv()

def generate_descriptions(excel_path: str) -> pd.DataFrame:
    """
    Liest die Excel-Datei ein und generiert Beschreibungen mit GPT-4.
    """
    # Excel-Datei einlesen
    df = pd.read_excel(excel_path, sheet_name="all CSPS")
    
    # ChatGPT-Modell initialisieren
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Liste für Beschreibungen
    descriptions = []
    uuids = []
    
    # Für jede Zeile eine Beschreibung generieren
    for _, row in df.iterrows():
        prompt = f"""
        Erstelle eine präzise, einzeilige Beschreibung auf Deutsch dieses Kriteriums eines Cloud-Service-Providers:
        Überbegriff: {row['Überbegriff']}
        Thema: {row['Thema']}
        Subthema: {row['Subthema']}
        
        Die Beschreibung soll kurz und prägnant sein.
        """
        
        response = llm.invoke(prompt)
        descriptions.append(response.content.strip())
        uuids.append(str(uuid.uuid4()))
    
    # Neue Spalten hinzufügen
    df.insert(0, 'UUID', uuids)
    df.insert(4, 'Beschreibung', descriptions)
    
    return df

def create_vector_database(df: pd.DataFrame, index_name: str):
    """
    Erstellt eine Pinecone Vektordatenbank aus dem DataFrame.
    """
    # Pinecone initialisieren mit neuer Syntax
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')
    )
    
    # Embedding-Modell initialisieren
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Wenn der Index nicht existiert, erstelle ihn
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,  # Dimension für OpenAI text-embedding-3-large
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Vektordatenbank erstellen
    texts = df['Beschreibung'].tolist()
    metadatas = df.drop(['Beschreibung'], axis=1).to_dict('records')
    
    vector_store = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=index_name
    )
    
    return vector_store


def convert_cloud_provider_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertiert die Cloud-Provider Spalten (AWS, GCP, Azure) zu booleschen Werten.
    'x' wird zu True, leere Werte oder andere Werte werden zu False.
    """
    cloud_providers = ['AWS', 'GCP', 'Azure']
    
    for provider in cloud_providers:
        df[provider] = df[provider].fillna('').str.lower() == 'x'
    
    return df


if __name__ == "__main__":
    # CSV-Datei lesen
    df_with_descriptions = pd.read_csv("matrix_with_descriptions.csv")
    
    # Vektordatenbank erstellen
    vector_store = create_vector_database(df_with_descriptions, "matrix-embeddings")
