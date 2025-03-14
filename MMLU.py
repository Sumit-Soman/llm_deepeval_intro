from transformers import pipeline
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

# Load the model using pipeline
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

# Function to generate responses
def generate_response(prompt):
    response = pipe(prompt, max_new_tokens=100, do_sample=True)
    return response[0]["generated_text"]

# Example test
messages = [{"role": "user", "content": "Who are you?"}]
response = generate_response(messages[0]["content"])
print("Model Response:", response)

# Define and evaluate MMLU benchmark
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=5
)

# Evaluate using the pre-trained model
benchmark.evaluate(model=pipe, batch_size=5)