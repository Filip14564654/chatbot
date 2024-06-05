import dspy
import dsp

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
retrieval_response = dsp.retrieve("When was the first FIFA World Cup held?", k=5)

for result in retrieval_response:
    print("Text:", result, "\n")

retrieval_response = colbertv2_wiki17_abstracts('When was the first FIFA World Cup held?', k=5)

for result in retrieval_response:
    print("Text:", result['text'], "\n")