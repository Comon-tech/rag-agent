
from crewai import Task
from textwrap import dedent
from preprocessor import PreProcess

preprocessor = PreProcess()

# tasks
class Tasks:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def __tip_section(self):
        return "If you do your BEST WORK, You'll be helping make the world a better place!"

    def pdf_task(self, agent, question):
        relevant_chunks = preprocessor.query_vector_store(question, self.vector_store)
        return Task(
            description=dedent(
                f"""
            get as munch information as fast as you can, retreived from {relevant_chunks}, that will be relevant to write the inception report
            {self.__tip_section()}
            Make sure to be as accurate as possible.
            """
            ),
            expected_output="Full data analysis.",
            agent=agent,
        )

    def writer_task(self, agent):
        return Task(
            description=dedent(
                f"""Take the input from task 1 and write about it.{self.__tip_section()}"""
            ),
            expected_output="make sure to include titles, descriptions, and examples where necesary",
            agent=agent,
        )
