from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import Annotated,List,Literal,TypedDict
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,BaseMessage
from src.main import graph



app=FastAPI()

class user_input(BaseModel):
    input:str


@app.post('/predict')
def input_pls(data:user_input):
    try:


        response=graph.invoke(
            {'messages':[HumanMessage(content=(data.input))]}
        )
        return {'response':response}
    except Exception as e:
        print(f"{e}")
        return None

    

        

