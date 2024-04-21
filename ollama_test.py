import dspy
import json

class ChatBot:
    def __init__(self, model_name):
        self.model = dspy.OllamaLocal(model=model_name)
        dspy.settings.configure(lm=self.model)
        self.qa_module = dspy.ChainOfThought('question -> answer')

    def ask_question(self, question):
        return self.qa_module(question=question).answer
    