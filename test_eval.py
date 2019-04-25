from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models
references = [["I am superman","Superman is me"], ["dog", "cat"]]
hypothesis = ["superman is me", "dog"]
print(nlgeval.compute_metrics(references, hypothesis))