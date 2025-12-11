from transformers import pipeline

nlp = pipeline("sentiment-analysis")
result = nlp("Saya sangat senang hari ini!")
print(result)
