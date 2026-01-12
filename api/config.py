import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

def get_env_variable(var_name: str, default: str = None) -> str:
    """Gets an environment variable or raises an error if it's not set."""
    value = os.getenv(var_name)
    if value is None:
        if default is None:
            raise ValueError(f"Environment variable '{var_name}' must be set.")
        return default
    return value

# --- API Keys ---
MURF_API_KEY = get_env_variable("MURF_API_KEY")
MURF_VOICE_ID = get_env_variable("MURF_VOICE_ID")
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")
PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = get_env_variable("PINECONE_INDEX_NAME", "mobile-phones")
PINECONE_EMBEDDING_DIMENSION = 3072

# --- LLM and Embeddings - Use a model with better multilingual support ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # More capable model for multilingual
    temperature=0.7,  # Slightly higher temperature for more creative responses
    google_api_key=GOOGLE_API_KEY
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)