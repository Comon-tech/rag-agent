from crewai import  Process, Crew
from agents import Agents
from tasks import Tasks

# the crew
class OurCrew:
    def __init__(self, question, vector_store):
        self.question = question
        self.vector_store = vector_store

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = Agents()
        tasks = Tasks(self.vector_store)

        # Define your custom agents and tasks here
        pdf_agent = agents.pdf_agent()
        writer_agent = agents.writer_agent()

        # Custom tasks include agent name and variables as input
        task1 = tasks.pdf_task(pdf_agent, self.question)

        task2 = tasks.writer_task(
            writer_agent,
        )

        # custom crew
        crew = Crew(
            agents=[pdf_agent, writer_agent],
            tasks=[task1, task2],
            verbose=True,
            process=Process.sequential,
            embedder={
                "provider": "cohere",
                "config": {"model": "embed-english-light-v3.0"},
            },
        )

        result = crew.kickoff()
        return result