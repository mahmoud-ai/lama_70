from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

print(llm(prompt="Explain to me the difference between Arabic union and Urobian union ?"))
