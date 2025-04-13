# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")