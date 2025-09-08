import os
import sys
from dotenv import load_dotenv # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain.schema.runnable import RunnableParallel # type: ignore

# Add the parent directory (two levels up) to the Python path
# This allows importing custom modules from the 'utils' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom functions to load different LLMs
from utils.load_llms import load_mistral, load_gemma

# Load environment variables from a .env file
load_dotenv()

# Load two different LLM models
model1 = load_mistral()  # Model 1: Mistral
model2 = load_gemma()    # Model 2: Gemma

# Prompt to generate short and simple notes from the input text
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# Prompt to generate 5 short question-answer pairs from the input text
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# Prompt to merge notes and quiz into a single document
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# Define an output parser that converts model outputs to plain strings
parser = StrOutputParser()

# RunnableParallel allows running multiple chains in parallel
# Here we generate 'notes' and 'quiz' at the same time using different models
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,  # Notes chain: prompt1 -> model1 -> parser
    'quiz': prompt2 | model2 | parser    # Quiz chain: prompt2 -> model2 -> parser
})

# Merge chain: combines notes and quiz into a single document
merge_chain = prompt3 | model1 | parser

# Complete chain: first run the parallel chain, then merge the outputs
chain = parallel_chain | merge_chain

# Input text on support vector machines
text = ''''
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
'''

# Invoke the chain with the input text
# Flow:
#   1. parallel_chain: generate notes (model1) and quiz (model2) simultaneously
#   2. merge_chain: combine the outputs into one document
result = chain.invoke({'text': text})

# Print the final merged document
print(result)

# Print an ASCII visualization of the chain structure for debugging
chain.get_graph().print_ascii()
