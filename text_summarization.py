from transformers import pipeline

summarizer = pipeline("summarization")
text = "Banyak teks panjang..."
print(summarizer(text)[0]["summary_text"])
