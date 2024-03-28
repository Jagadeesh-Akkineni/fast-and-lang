import asyncio
from functools import lru_cache
from typing import AsyncGenerator
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from pydantic import BaseModel 
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Settings class for this application.
    Utilizes the BaseSettings from pydantic for environment variables.
    """

    groq_api_key: str
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    """Function to get and cache settings.
    The settings are cached to avoid repeated disk I/O.
    """
    return Settings()


class StreamingConversationChain:
    """
    Class for handling streaming conversation chains.
    It creates and stores memory for each conversation,
    and generates responses using the ChatOpenAI model from LangChain.
    """

    def __init__(self, groq_api_key: str, temperature: float = 0.0):
        self.memories = {}
        self.groq_api_key = groq_api_key
        self.temperature = temperature

    async def generate_response(
        self, conversation_id: str, message: str
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous function to generate a response for a conversation.
        It creates a new conversation chain for each message and uses a
        callback handler to stream responses as they're generated.

        :param conversation_id: The ID of the conversation.
        :param message: The message from the user.
        """
        callback_handler = AsyncIteratorCallbackHandler()
        llm = ChatGroq(
            callbacks=[callback_handler],
            streaming=True,
            temperature=self.temperature,
            groq_api_key=self.groq_api_key,
        )

        memory = self.memories.get(conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self.memories[conversation_id] = memory

        chain = ConversationChain(
            memory=memory,
            prompt=CHAT_PROMPT_TEMPLATE,
            llm=llm,
        )

        run = asyncio.create_task(chain.ainvoke(input=message))

        async for token in callback_handler.aiter():
            yield token

        await run, print(type(run)), print(run)

        # Assuming run is the completed asyncio task
        result = run.result()

        # Access the 'history' key from the result dictionary
        history = result['history']

        # Iterate through the messages in history
        for message in history:
            # Check if the message is of type AIMessage
            # if isinstance(message):
                # Print the content of the AIMessage
                print(message.content)

            # Writing the formatted messages to the text file
                with open("test.txt", "a+") as file:
                    file.truncate(0)
                    file.write(message.content)


class ChatRequest(BaseModel):
    """Request model for chat requests.
    Includes the conversation ID and the message from the user.
    """

    conversation_id: str
    message: str


CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You're a AI that knows everything about cats."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

app = FastAPI(dependencies=[Depends(get_settings)])

streaming_conversation_chain = StreamingConversationChain(
    groq_api_key=get_settings().groq_api_key
)


@app.post("/chat", response_class=StreamingResponse)
async def generate_response(data: ChatRequest) -> StreamingResponse:
    """Endpoint for chat requests.
    It uses the StreamingConversationChain instance to generate responses,
    and then sends these responses as a streaming response.

    :param data: The request data.
    """
    return StreamingResponse(
        streaming_conversation_chain.generate_response(
            data.conversation_id, data.message
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)