# SPDX-FileCopyrightText: ASSUME Developers
# SPDX-License-Identifier: AGPL-3.0-or-later

# import requests
import os
from mistralai import Mistral

import os
import requests

class LLMModelAPI:
    def __init__(self, model="granite3.1-dense:8b", api_url="http://localhost:8080/api/chat/completions"):
        self.api_url = api_url
        self.model = model

        # Get API token from environment variable
        self.token = os.environ.get("FHAACHEN_API_TOKEN")
        if not self.token:
            raise ValueError("FHAACHEN_API_TOKEN environment variable not set.")

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def query(self, prompt: str) -> str:
        """Sends a prompt to the Granite API and returns the response text."""
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()

        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()


# Mistral paid api
# class LLMModelAPI:
#     def __init__(self, model="mistral-small-latest"):
#         api_key = os.environ.get("MISTRAL_API_KEY")
#         if not api_key:
#             raise ValueError("MISTRAL_API_KEY environment variable not set.")
        
#         self.client = Mistral(api_key=api_key)
#         self.model = model

#     def query(self, prompt: str) -> str:
#         """Sends a prompt to the Mistral API and returns the text output."""
#         chat_response = self.client.chat.complete(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#         )
#         # Return the content of the first choice's message
#         return chat_response.choices[0].message.content.strip()






    ### LM studio, local implementation
    # """
    # Encapsulates the logic for interacting with an LLM API.
    # Handles URL, headers, payload formatting, and response extraction.
    # """

    # def __init__(self, api_url="http://localhost:1234/v1/completions", model="Mistral-7B-Instruct-v0.3-Q4_K_M", max_tokens=4000):
    #     self.api_url = api_url
    #     self.model = model
    #     self.max_tokens = max_tokens
    #     self.headers = {"Content-Type": "application/json"}

    # def query(self, prompt: str) -> str:
    #     """Sends a prompt to the LLM API and returns the raw text output."""
    #     payload = {
    #         "model": self.model,
    #         "prompt": prompt,
    #         "max_tokens": self.max_tokens
    #     }

    #     response = requests.post(self.api_url, headers=self.headers, json=payload)
    #     response.raise_for_status()

    #     return response.json().get("choices", [{}])[0].get("text", "").strip()
