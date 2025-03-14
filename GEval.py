from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.dataset import EvaluationDataset

golden_actual_text = """
Quantum entanglement is a phenomenon in quantum physics where two or more particles become linked, 
such that the state of one particle instantly influences the state of the other, no matter how far apart they are. 
This connection persists even if the particles are separated by large distances.
"""

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o",  # Define the evaluation model
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Check if the 'actual output' contains any factual inaccuracies compared to 'expected output'.",
        "Lightly penalize omission of key details but prioritize correctness of core concepts.",
        "Minor rewordings and simplifications are acceptable as long as the main idea is preserved."
    ],
)

# Define test cases with different LLM-generated responses
test_case_1 = LLMTestCase(
    input="Explain the concept of quantum entanglement in simple terms.",
    actual_output="Quantum entanglement happens when two particles stay connected, so a change in one affects the other instantly, even miles apart.",
    expected_output=golden_actual_text
)

test_case_2 = LLMTestCase(
    input="Explain the concept of quantum entanglement in simple terms.",
    actual_output="Quantum entanglement is when two particles always have the same properties, even at a distance, due to hidden variables in physics.",
    expected_output=golden_actual_text
)

test_case_3 = LLMTestCase(
    input="Explain the concept of quantum entanglement in simple terms.",
    actual_output="Quantum entanglement is the process by which atoms travel faster than light and communicate instantly across space.",
    expected_output=golden_actual_text
)

# Create an evaluation dataset
dataset = EvaluationDataset(test_cases=[test_case_1, test_case_2, test_case_3])

# Run evaluation
evaluation_output = dataset.evaluate([correctness_metric])

# Print the evaluation results
print(evaluation_output)