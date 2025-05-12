from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)
print(prompt)
print(prompt.format(topic="sports", language="spanish"))

print("=========")

prompt = SystemMessage(content="You are a nice pirate")
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)
print(new_prompt.format_messages(input="i said hi"))
print(new_prompt.format(input="i said hi"))