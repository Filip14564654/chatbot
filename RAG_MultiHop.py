import dspy
from dspy.teleprompt import BootstrapFewShot
from dsp.utils import deduplicate

import pandas as pd
from dspy.datasets.dataset import Dataset




ollamallama3 = dspy.OllamaLocal(model='llama3')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=ollamallama3, rm=colbertv2_wiki17_abstracts)

#split csv data, self._train split first x rows into train data and so do self._dev
class CSVDataset(Dataset):
    def __init__(self, file_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        df = pd.read_csv(file_path)
        self._train = df.iloc[0:40].to_dict(orient='records')

        self._dev = df.iloc[40:].to_dict(orient='records')

dataset = CSVDataset("train_data.csv")
#print(dataset.train[:3])


# Load the of training and validation data
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

#template for generating answears
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 30 words")

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

#define how task will be proceed, num_passages is number of context for question
#self.retrieve load of relevant contexts
#self.generate_answear allow you to generate answear with defined structure (signature)
#method forward is for generation of answear where it take question with context, then make prediction and give us an answear
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    
# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)

# Compile!
compiled_baleen = teleprompter.compile(SimplifiedBaleen(), trainset=trainset)

# Ask any question you like to this simple RAG program.
my_question = "Jaký je účel přednášek na Ostravské univerzitě a jaký údaj je přidělen studujícímu po registraci do elektronického informačního systému na OU?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
pred = uncompiled_baleen(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
