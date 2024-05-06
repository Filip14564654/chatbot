import dspy
from dspy.teleprompt import BootstrapFewShot

import pandas as pd
from dspy.datasets.dataset import Dataset



ollamallama3 = dspy.OllamaLocal(model='llama3', max_tokens=4000, timeout_s=480)
#ollamallama3 = dspy.OllamaLocal(model="llama3:8b-instruct-q5_1", max_tokens=4000, timeout_s=480)
#ollamallama3 = dspy.OllamaLocal(model='mistral')
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

#define how task will be proceed, num_passages is number of context for question
#self.retrieve load of relevant contexts
#self.generate_answear allow you to generate answear with defined structure (signature)
#method forward is for generation of answear where it take question with context, then make prediction and give us an answear
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    
# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

# Ask any question you like to this simple RAG program.
my_question = "Jaký je účel přednášek na Ostravské univerzitě a kdo zodpovídá za odbornou úroveň daného předmětu?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")
