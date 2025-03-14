from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

context = [
    "OpenAI was founded in December 2015 as an artificial intelligence research organization.",
    "Its primary goal is to ensure AI benefits all of humanity.",
    "ChatGPT is one of OpenAI’s most popular products, launched in November 2022."
]

# AI Model’s Response (Potential Hallucination)
actual_output = "OpenAI was founded in 2012 and initially focused on robotics before pivoting to AI research."

# Define the test case
test_case = LLMTestCase(
    input="When was OpenAI founded and what was its focus?",
    actual_output=actual_output,
    context=context
)

# Define the hallucination metric
metric = HallucinationMetric(threshold=0.3)

# Evaluate the response
evaluate(test_cases=[test_case], metrics=[metric])