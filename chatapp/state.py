import os

from openai import AsyncOpenAI
from dotenv import load_dotenv
import reflex as rx

load_dotenv()


class State(rx.State):
    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    def format_messages(self):
        # Format chat history into the required message structure
        messages = []

        # Add previous chat history
        for question, answer in self.chat_history:
            # Add user message
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": question}]}
            )

            # Add assistant message
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )

        # Add the current question
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": self.question}]}
        )

        return messages

    async def answer(self):
        # Our chatbot has some brains now!
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Get formatted message history
        messages = self.format_messages()

        session = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stop=None,
            temperature=0.7,
            stream=True,
        )

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        async for item in session:
            if hasattr(item.choices[0].delta, "content"):
                if item.choices[0].delta.content is None:
                    # presence of 'None' indicates the end of the response
                    break
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield
