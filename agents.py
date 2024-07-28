from crewai import Agent
from config import llm
from textwrap import dedent

# agents
class Agents:
    def __init__(self):
        # Initialize language model
        self.llm = llm

    def pdf_agent(self):
        return Agent(
            role="Geologist",
            backstory=dedent(
                f"""You work in the energy and mining sectors to exploit natural resources"""
            ),
            goal=dedent(
                f"""Uncover any information from the quatation request document provided."""
            ),
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def writer_agent(self):
        return Agent(
            role="Writer",
            backstory=dedent(
                f"""You are good at writing reports and explaining and writing their summaries."""
            ),
            goal=dedent(
                f"""Make sure you understand the question provided by the user."""
            ),
            verbose=True,
            llm=self.llm,
            allow_delegation=False,
        )