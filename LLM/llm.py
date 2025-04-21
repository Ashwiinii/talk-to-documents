from typing import List
from openai import OpenAI
from LLM.prompt import RAG_PROMPT_TEMPLATE
import streamlit as st


class LLMInterface:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    def query(
        self,
        retrieved_responses: List[str],
        user_query: str,
        llm_name: str = "phi4",
        system_role: str = "You are a helpful assistant.",
        temperature: float = 0.0,
    ) -> str:
        context_str = (
            "\n".join(retrieved_responses)
            if retrieved_responses
            else "No relevant context found."
        )
        prompt = RAG_PROMPT_TEMPLATE.format(question=user_query, context=context_str)

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=llm_name, messages=messages, temperature=temperature
            )
            response = completion.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            response = "API call encountered an error."

        return response
