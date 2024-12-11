from summarization import TextSummarizer
import nltk

# Ensure the 'punkt' tokenizer is downloaded
nltk.download('punkt')

# Sample text
text = """
Artificial Intelligence (AI) is a rapidly growing field that is transforming industries worldwide.
From healthcare to education, AI is automating processes and improving efficiency. It is used in
personalized recommendations, autonomous vehicles, and fraud detection. However, with great power
comes great responsibility. The ethical implications of AI, such as bias and privacy concerns, must
be addressed to ensure its safe and fair use. As the technology advances, it opens up new
possibilities, but also challenges us to think critically about its impact on society.
"""

# Initialize the summarizer
summarizer = TextSummarizer()

# Summarize the text
summary = summarizer.summarize(text, n=3)
print("Original Text:")
print(text)
print("\nSummarized Text:")
print(summary)
