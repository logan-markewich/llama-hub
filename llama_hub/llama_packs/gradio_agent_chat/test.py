from .base import GradioAgentChatPack
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

if __name__ == "__main__":
    agent = ReActAgent.from_tools(
        tools=[multiply_tool, add_tool], llm=OpenAI(), verbose=True
    )
    pack = GradioAgentChatPack(agent=agent)
    pack.run()