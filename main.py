"""
How to Build a Simple ChatBot with Langchain.

This is a planned tutorial of making chatbot and agentic ai with langchain.
In this perticular tutorial we are make a very basic coversational bot.
Later in the series it will receive more add-on.

Blog:- https://blog.chatembedai.com/series/build-ai-chatbots
"""

import uuid
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

""" load environment variables """
load_dotenv()


class ChatBot:
    """
    ChatBot class used to chat with a model.

    Attributes
    ----------
    model : string
        The LLM model to use during chat
    model_provider: string
        The provider who provides this LLM

    Methods
    -------
    chat_with_me():
        Returns the LLM response.
    """

    def __init__(self, model: str, model_provider: str, session_id: str | None) -> None:
        """
        Constructs all the necessary attributes for the Chat.

        Parameters
        ----------
        model : string
            The LLM model to use during chat
        model_provider: string
            The provider who provides this LLM
        session_id : string | None
            Unique session id for the chat session
        """
        self.model = model
        self.model_provider = model_provider
        self.__initialize_model()
        self.memory = InMemorySaver()
        self.session = session_id if session_id else str(uuid.uuid4())

    def __initialize_model(self) -> None:
        """
        Private method to set the chat model
        """
        self.chat_model = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=0.5,
            timeout=10,
        )

    def __get_agent(self) -> Any:
        """
        Private method to create the agent
        """
        return create_agent(model=self.chat_model, checkpointer=self.memory)

    def chat_with_me(self, message: str) -> str:
        """
        This method helps with LLM chat.

        Attributes
        ----------
        message : string
            User message that will be processed by the LLM

        Returns
        -------
        string
            The LLM response
        """
        try:
            agent = self.__get_agent()
            human_message = HumanMessage(content=message)

            response = agent.invoke(
                {"messages": [human_message]},
                {"configurable": {"thread_id": self.session}},
            )

            return response["messages"][-1].content
        except Exception as e:
            return f"Error Occured: {str(e)}"


if __name__ == "__main__":
    try:
        chatbot = ChatBot("gpt-4o-mini", "openai", "example_session_1")
        print("ChatBot initialized. Type 'q' to quit.")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "q":
                print("Goodbye!")
                break

            if user_input:
                response = chatbot.chat_with_me(user_input)
                print(f"Bot: {response}")
            else:
                print("Please enter a message.")
    except KeyboardInterrupt:
        print("\nGoodbye!")
