from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

#llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

response = llm.invoke("Who invented Python? Answer in one sentence.")

print(response.content)

