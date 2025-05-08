from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage('Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage("Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"),
    HumanMessage("what do you call a speechless parrot")
]

result = trim_messages(
    messages,
    # Keep the last <= n_count tokens of the messages.
    strategy="last",
    token_counter=len,
    # When token_counter=len, each message
    # will be counted as a single token.
    # Remember to adjust based on the desired conversation length
    max_tokens=5,
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    start_on="human",
    # Most chat models expect that chat history ends with either:
    # (1) a HumanMessage or
    # (2) a ToolMessage
    end_on=("human", "tool"),
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True
)

print(result)