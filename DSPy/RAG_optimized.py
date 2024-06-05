import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
import pandas as pd
from dspy.datasets.dataset import Dataset

# Nastavení retrieval a language modelů
ollamallama3 = dspy.OllamaLocal(model='llama3', max_tokens=4000, timeout_s=480)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=ollamallama3, rm=colbertv2_wiki17_abstracts)

# Dataset pro načítání CSV dat
class CSVDataset(Dataset):
    def __init__(self, file_path, train_size=100, dev_size=50, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        df = pd.read_csv(file_path)
        self._train = df.iloc[0:train_size].to_dict(orient='records')
        self._dev = df.iloc[train_size:train_size + dev_size].to_dict(orient='records')

dataset = CSVDataset("train_data.csv", train_size=100, dev_size=50)

# Načtení tréninkových a validačních dat
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# Definice šablony pro generování odpovědí
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    fallback = dspy.InputField(default="I'm sorry, I don't have enough information to answer that.")
    answer = dspy.OutputField(desc="often between 1 and 30 words")

# Šablona pro dynamický fallback
class GenerateDynamicFallback(dspy.Signature):
    """Generate fallback answers dynamically based on the question."""

    question = dspy.InputField()
    fallback = dspy.OutputField(desc="A fallback answer for cases with no relevant context")

# Vylepšená třída RAG
class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()

        # Inicializace Retrieval modulu s více kroky
        self.retrieve_primary = dspy.Retrieve(k=num_passages)
        self.retrieve_secondary = dspy.Retrieve(k=2 * num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.dynamic_fallback = dspy.ChainOfThought(GenerateDynamicFallback)

    def forward(self, question):
        # Primární retrieval krok
        retrieved_primary = self.retrieve_primary(question)

        if retrieved_primary.passages:
            context = retrieved_primary.passages
            fallback = ""
        else:
            # Sekundární retrieval krok jako záložní
            retrieved_secondary = self.retrieve_secondary(question)
            context = retrieved_secondary.passages if retrieved_secondary.passages else []
            fallback = self.dynamic_fallback(question).fallback

        # Generování odpovědi nebo fallbacku
        prediction = self.generate_answer(context=context, question=question, fallback=fallback)
        return dspy.Prediction(context=context, answer=prediction.answer)

# Validační funkce: Kontrola správnosti odpovědi a relevantního kontextu
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return float(answer_EM and answer_PM)

# Nastavení optimalizéru BootstrapFewShotWithRandomSearch
config = dict(max_bootstrapped_demos=3, max_labeled_demos=3, num_candidate_programs=10, num_threads=1)

teleprompter = BootstrapFewShotWithRandomSearch(metric=validate_context_and_answer, **config)

# Kompilace RAG programu pomocí optimalizéru
optimized_program = teleprompter.compile(RAG(), trainset=trainset)

# Dotaz na otázku pomocí optimalizovaného RAG modulu
my_question = "Jaký je účel přednášek na Ostravské univerzitě? A co se stane, když student neprovede zápis předmětů ve stanoveném termínu a předepsaným způsobem?"
pred = optimized_program(my_question)

# Tisk kontextu a odpovědi
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

# Set up the evaluator, which can be re-used in your code.
evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

# Launch evaluation.
evaluator(optimized_program, metric=validate_context_and_answer)
