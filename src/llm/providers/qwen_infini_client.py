# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from typing import Any, Dict, List

from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from src.llm.provider_client_base import LLMProviderClientBase

from src.logging.logger import bootstrap_logger

import os

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")

logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class QwenInfiniClient(LLMProviderClientBase):
    """Qwen LLM Provider Client for Infinigence AI (cloud.infini-ai.com)

    This client supports Qwen models deployed on the Infinigence AI platform,
    using OpenAI-compatible API with tool calling support.
    """

    def _create_client(self, config: DictConfig):
        """Create configured Qwen client using Infinigence AI endpoint"""
        # Use infini_api_key and infini_base_url from config
        api_key = self.cfg.llm.get("infini_api_key") or self.cfg.llm.get("qwen_api_key")
        base_url = self.cfg.llm.get("infini_base_url") or self.cfg.llm.get("qwen_base_url")

        if not api_key:
            raise ValueError("INFINI_API_KEY or QWEN_API_KEY must be set in config")
        if not base_url:
            raise ValueError("INFINI_BASE_URL or QWEN_BASE_URL must be set in config")

        if self.async_client:
            return AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=1800,
            )
        else:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=1800,
            )

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    async def _create_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
    ):
        """
        Send message to Qwen API (Infinigence AI platform).

        :param system_prompt: System prompt string.
        :param messages: Message history list.
        :param tools_definitions: Tool definitions for function calling.
        :param keep_tool_result: Number of tool results to keep in history.
        :return: Qwen API response object or None (if error occurs).
        """
        logger.debug(f"Calling Qwen LLM ({'async' if self.async_client else 'sync'})")

        # Add system prompt to messages
        if system_prompt:
            # Check if there's already a system message
            if messages and messages[0]["role"] == "system":
                # Replace existing message
                messages[0] = {
                    "role": "system",
                    "content": system_prompt,
                }
            else:
                # Insert new message
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

        messages_copy = self._remove_tool_result_from_messages(
            messages, keep_tool_result
        )

        tool_list = await self.convert_tool_definition_to_tool_call(tools_definitions)

        try:
            params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": messages_copy,
                "tools": tool_list,
                "stream": False,
            }

            # Add optional parameters
            if self.top_p != 1.0:
                params["top_p"] = self.top_p
            # NOTE: min_p and top_k may be supported by vLLM-based deployments
            if self.min_p != 0.0:
                params["min_p"] = self.min_p
            if self.top_k != -1:
                params["top_k"] = self.top_k
            if self.repetition_penalty != 1.0:
                params["repetition_penalty"] = self.repetition_penalty

            response = await self._create_completion(params, self.async_client)

            logger.debug(
                f"Qwen LLM call status: {getattr(response.choices[0], 'finish_reason', 'N/A')}"
            )
            return response
        except asyncio.CancelledError:
            logger.exception("[WARNING] Qwen LLM API call was cancelled during execution")
            raise
        except Exception as e:
            logger.exception(f"Qwen LLM call failed: {str(e)}")
            raise e

    async def _create_completion(self, params: Dict[str, Any], is_async: bool):
        """Helper to create a completion, handling async and sync calls."""
        if is_async:
            return await self.client.chat.completions.create(**params)
        else:
            return self.client.chat.completions.create(**params)

    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool]:
        """Process Qwen LLM response

        Qwen uses OpenAI-compatible response format with tool_calls.
        """

        if not llm_response or not llm_response.choices:
            error_msg = "Qwen LLM did not return a valid response."
            logger.debug(f"Error: {error_msg}")
            return "", True  # Exit loop

        # Extract LLM response text
        finish_reason = llm_response.choices[0].finish_reason

        if finish_reason == "stop":
            assistant_response_text = llm_response.choices[0].message.content or ""
            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )
        elif finish_reason == "tool_calls":
            # For tool_calls, we need to extract tool call information
            tool_calls = llm_response.choices[0].message.tool_calls
            assistant_response_text = llm_response.choices[0].message.content or ""

            # If there's no text content, we generate a text describing the tool call
            if not assistant_response_text:
                tool_call_descriptions = []
                for tool_call in tool_calls:
                    tool_call_descriptions.append(
                        f"Using tool {tool_call.function.name} with arguments: {tool_call.function.arguments}"
                    )
                assistant_response_text = "\n".join(tool_call_descriptions)

            message_history.append(
                {
                    "role": "assistant",
                    "content": assistant_response_text,
                    "tool_calls": [
                        {
                            "id": _.id,
                            "type": "function",
                            "function": {
                                "name": _.function.name,
                                "arguments": _.function.arguments,
                            },
                        }
                        for _ in tool_calls
                    ],
                }
            )
        elif finish_reason == "length":
            assistant_response_text = llm_response.choices[0].message.content or ""
            if assistant_response_text == "":
                assistant_response_text = "Qwen LLM response is empty. This is likely due to context length limit."
            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )
        else:
            logger.warning(
                f"Unsupported finish reason: {finish_reason}, treating as stop"
            )
            assistant_response_text = llm_response.choices[0].message.content or ""
            message_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )

        logger.debug(f"Qwen LLM Response: {assistant_response_text}")

        return assistant_response_text, False

    def extract_tool_calls_info(self, llm_response, assistant_response_text):
        """Extract tool call information from Qwen LLM response

        Qwen uses OpenAI-compatible tool_calls format.
        """
        from src.utils.parsing_utils import parse_llm_response_for_tool_calls

        # For Qwen, get tool calls directly from response object (OpenAI-compatible)
        if llm_response.choices[0].finish_reason == "tool_calls":
            return parse_llm_response_for_tool_calls(
                llm_response.choices[0].message.tool_calls
            )
        else:
            return [], []

    def update_message_history(
        self, message_history, tool_call_info, tool_calls_exceeded: bool = False
    ):
        """Update message history with tool calls data (llm client specific)

        Qwen uses OpenAI-compatible format with 'tool' role.
        """

        for cur_call_id, tool_result in tool_call_info:
            message_history.append(
                {
                    "role": "tool",
                    "tool_call_id": cur_call_id,
                    "content": tool_result["text"],
                }
            )

        return message_history

    def handle_max_turns_reached_summary_prompt(self, message_history, summary_prompt):
        """Handle max turns reached summary prompt"""
        return summary_prompt
