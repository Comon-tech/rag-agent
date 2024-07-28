
from textwrap import dedent
from preprocessor import PreProcess
from crew import OurCrew

import  logging

preprocessor = PreProcess()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store = preprocessor.store_embeddings(file_path="Bultfonteinite.pdf")
# vector_store
print("## Comon AI Agents ####")
print("-------------------------------")
question = input(dedent("""Enter your question: """))

custom_crew = OurCrew(question, vector_store)
result = custom_crew.run()
