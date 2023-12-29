"""
Title:  Getting Started with LangChain: A Beginner’s Guide to Building LLM-Powered Applications
            - A LangChain tutorial to build anything with large language models in Python
Author: Leonie Monigatti
Source: https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c#bd03

"""

# imports 
import os
import langchain
from langchain.llms import OpenAI
from langchain import ConversationChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.agents import AgentType, load_tools, initialize_agent
from langchain.chains import LLMChain, PromptTemplate, SimpleSequentialChain, RetrievalQA


# OpenAI API Key
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key



# getting a simple response 
# --------------------------
llm = OpenAI(temperature=0.6)
name = llm("I want to open an indian restarant. Suggest a name for this")
print(name)


# similar but response based on prompt using different model 
# (NOTE:text-davinci-003 cost more per API request)
# -----------------------------------------------------------
llm = OpenAI(model_name="text-davinci-003")
prompt = "Alice has a parrot. What animal is Alice's pet?"
completion = llm(prompt)
print(completion)


# chaining prompt and template
# -----------------------------
llm = OpenAI()
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(input_variables=["product"], template=template)
chain = LLMChain(llm = llm, prompt = prompt)
result = chain.run("vibrant t-shirts")



# chaining multiple prompts and templates 
# ----------------------------------------------------
first_prompt = PromptTemplate(input_variables=["product"], template="What is a good name for a company that makes {product}?")
second_prompt = PromptTemplate(input_variables=["company_name"], template="Write a catchphrase for the following company: {company_name}")

chain_one = LLMChain(llm = llm, prompt = first_prompt)
chain_two = LLMChain(llm=llm, prompt= second_prompt)

# # Combine the first and the second chain 
overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

# # Run the chain specifying only the input variable for the first chain.
catchphrase = overall_chain.run("vibrant t-shirts")
print(catchphrase)

"""
--Output Sample 1--
# > Entering new SimpleSequentialChain chain...
# Vivacious Tees.
# "Be Vivacious with Vivacious Tees!"
# > Finished chain.

--Output Sample 2--
> Entering new SimpleSequentialChain chain...
"Colorburst Apparel"
"Bursting with Color, Made for You"
> Finished chain.
"""

# Answering questions based on provided data (youtube subtitle) 
# -------------------------------------------------------------- 
embeddings = OpenAIEmbeddings()
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=g1pb2aK2we4")
documents = loader.load()

# # create the vectorestore to use as the index
db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)

query = "What are the two kinds of bridges in the world?"
print(query)
result = qa({"query": query})
print(result)

"""
--sample 1--
query: 'What are the two kinds of streches atheletes do?'
result: 'The two kinds of stretches that athletes do are dynamic stretches and static stretches.'
--sample 2--
query: 'What are the two kinds of bridges in the world?'
result': 'I do not know.'
"""


# Conversation 
# -------------
conversation = ConversationChain(llm=llm, verbose=True)
output1 = conversation.predict(input="Alice has a parrot.")
output2 = conversation.predict(input="Bob has two cats.")
output3 = conversation.predict(input="How many pets do Alice and Bob have?")
print("Output 3:", output3)

"""
Output 3:  I don't have enough information to accurately answer that question. But based on what you've told me, I know that 
Alice has at least one parrot and Bob has at least two cats. It's possible that they have more pets that we don't know about.
"""


# Agents
# ----------------
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True)


agent.run("When was the movie back to the future released? How old is it in 2024?")

"""
> Entering new AgentExecutor chain...

I should use Wikipedia to find the answer.
Action: Wikipedia
Action Input: "Back to the Future"
Observation: Page: Back to the Future
Summary: Back to the Future is a 1985 American ...... prompted numerous books about its production, documentaries, and commercials.

Page: Back to the Future (franchise)
Summary: Back to the Future is an American science fiction comedy ..... as well as a video game and a stage 
Thought: I now know the final answer.
Final Answer: The movie Back to the Future was released on July 3, 1985.

> Finished chain.
The movie Back to the Future was released on July 3, 1985.

> Entering new AgentExecutor chain...
 I need to find out the year in question and subtract it from 2024.
Action: Calculator
Action Input: 2024 - 1985
Observation: Answer: 39
"""