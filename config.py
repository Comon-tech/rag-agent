import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from decouple import config
from dotenv import load_dotenv

load_dotenv()

embeding_model_name="models/embedding-001"
# initialize the gemini model
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                            verbose=True,
                            temperature=0.5,
                            goggle_api_key=os.getenv('GOOGLE_API_KEY'))
embeddings_model = GoogleGenerativeAIEmbeddings(model=embeding_model_name)