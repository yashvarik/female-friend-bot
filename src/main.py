from langgraph.graph import StateGraph,START,END,add_messages
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage,BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode,tools_condition
from typing_extensions import Annotated,List,Literal,TypedDict
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path='femalebot.db'))


load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
tavily_api_Key=os.getenv("TAVILY_API_KEY")

search_tool=TavilySearch(
    
    max_results=1
)


tools=[search_tool]

tool_node=ToolNode(tools=tools)



llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0,
    
)

llm_with_tool=llm.bind_tools(tools)

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    




def chat(state:State):
    messages=state["messages"][-1].content
    response=llm_with_tool.invoke([
        SystemMessage(content=(
            """ You are female friend bot desinged to help the men to ask SENSETIVE QUESTIONS regarding WOMENS.
            You have to guide the user who will be a men .
            They will ask you questions regarding womens , you have to answer it as  their women friend.
            Remember keep the tone friendly and helping.
            Don't answer any eriotic questions.
            you will help the men user to discuss some issues regarding womens.
             You have the access of the search tool , So use it when ever need it.
             You can answer questions , regarding SEX EDUCATION,but not the ERIOTICA.
             At end of the Answer always display any random facts about womens so it feel little personal and friendly."""
        )),
        HumanMessage(content=(messages))
    ])
    return {'messages':[response]}




q=StateGraph(State)
q.add_node('chat',chat)
q.add_node('tools',tool_node)

q.add_edge(START,'chat')
q.add_conditional_edges('chat',tools_condition)
q.add_edge('tools','chat')

graph=q.compile()

graph


if __name__ == "__main__":
    for msg,metadata in graph.stream({
        'messages':'how long does the period last in girls '
    },stream_mode='messages'):
        print(msg.content,end='',flush=True)


    
