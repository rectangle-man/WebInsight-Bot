import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import HuggingFaceHub



# from getpass import getpass


HUGGINGFACEHUB_API_TOKEN = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "google/flan-t5-large"

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature":0, "max_length":180, 'max_new_tokens' : 120, 'top_k' : 10, 'top_p': 0.95, 'repetition_penalty':1.03}
)

llm_chain = LLMChain(prompt = prompt, llm = llm)
                     
question = "" #contains the question

print(llm_chain.invoke({"question": question}))




