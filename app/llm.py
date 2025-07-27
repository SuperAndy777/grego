import math
from typing import Dict, List, Optional, Union

import litellm
import tiktoken
from openai import APIError, AuthenticationError, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

# --- Constants ---
REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.5-pro-latest", # Added gemini model
]


# --- TokenCounter Class (from your original file) ---
class TokenCounter:
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        detail = image_item.get("detail", "medium")
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS
        if detail in ["high", "medium"] and "dimensions" in image_item:
            width, height = image_item["dimensions"]
            return self._calculate_high_detail_tokens(width, height)
        return self._calculate_high_detail_tokens(1024, 1024)

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width, height = int(width * scale), int(height * scale)
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width, scaled_height = int(width * scale), int(height * scale)
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        return (tiles_x * tiles_y * self.HIGH_DETAIL_TILE_TOKENS) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        if not content: return 0
        if isinstance(content, str): return self.count_text(content)
        token_count = 0
        for item in content:
            if isinstance(item, str): token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item: token_count += self.count_text(item["text"])
                elif "image_url" in item: token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        total_tokens = self.FORMAT_TOKENS
        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS
            tokens += self.count_text(message.get("role", ""))
            if "content" in message: tokens += self.count_content(message["content"])
            if "tool_calls" in message: tokens += self.count_tool_calls(message["tool_calls"])
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))
            total_tokens += tokens
        return total_tokens


# --- LLM Class (Corrected and Restored) ---
class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        if hasattr(self, "tokenizer"): return

        llm_config = llm_config or config.llm
        llm_config = llm_config.get(config_name, llm_config["default"])

        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_type = llm_config.api_type
        self.api_key = llm_config.api_key
        self.api_version = llm_config.api_version
        self.base_url = llm_config.base_url
        self.max_input_tokens = getattr(llm_config, "max_input_tokens", None)

        self.total_input_tokens = 0
        self.total_completion_tokens = 0

        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.token_counter = TokenCounter(self.tokenizer)

    def get_litellm_model_name(self):
        """Constructs the model name string for LiteLLM."""
        if self.api_type == "azure": return f"azure/{self.model}"
        if self.api_type == "google": return f"gemini/{self.model}"
        return self.model

    async def _litellm_completion(self, **kwargs):
        """A single, reusable method to call litellm.acompletion."""
        params = {
            "model": self.get_litellm_model_name(),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if "temperature" not in params:
            params["temperature"] = self.temperature
        return await litellm.acompletion(**params)

    @staticmethod
    def _clean_tools_for_gemini(obj):
        """Recursively removes the 'dependencies' key from tool parameters, as it's not supported by Gemini."""
        if isinstance(obj, dict):
            return {k: LLM._clean_tools_for_gemini(v) for k, v in obj.items() if k != 'dependencies'}
        elif isinstance(obj, list):
            return [LLM._clean_tools_for_gemini(item) for item in obj]
        else:
            return obj

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count_text(text)

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        if self.max_input_tokens is not None and (self.total_input_tokens + input_tokens) > self.max_input_tokens:
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"
        return "Token limit exceeded"

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]], supports_images: bool = False) -> List[dict]:
        formatted_messages = []
        for message in messages:
            if isinstance(message, Message): message = message.to_dict()
            if isinstance(message, dict):
                if "role" not in message: raise ValueError("Message dict must contain 'role' field")
                if supports_images and message.get("base64_image"):
                    if not message.get("content"): message["content"] = []
                    elif isinstance(message["content"], str): message["content"] = [{"type": "text", "text": message["content"]}]
                    elif isinstance(message["content"], list):
                        message["content"] = [{"type": "text", "text": item} if isinstance(item, str) else item for item in message["content"]]
                    message["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{message['base64_image']}"}})
                    del message["base64_image"]
                elif not supports_images and message.get("base64_image"):
                    del message["base64_image"]
                
                # ** THE FIX IS HERE **
                # Ensure message has content before adding it.
                if message.get("content") or message.get("tool_calls"):
                    formatted_messages.append(message)
            else: raise TypeError(f"Unsupported message type: {type(message)}")
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES: raise ValueError(f"Invalid role: {msg['role']}")
        return formatted_messages

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type((OpenAIError, Exception, ValueError)))
    async def ask_tool(self, messages: List[Union[dict, Message]], system_msgs: Optional[List[Union[dict, Message]]] = None, timeout: int = 300, tools: Optional[List[dict]] = None, tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO, temperature: Optional[float] = None, **kwargs) -> ChatCompletionMessage | None:
        try:
            if not messages and not system_msgs:
                logger.warning("ask_tool was called with no messages. Aborting.")
                return None

            if tool_choice not in TOOL_CHOICE_VALUES: raise ValueError(f"Invalid tool_choice: {tool_choice}")
            supports_images = self.model in MULTIMODAL_MODELS
            
            all_msgs = (system_msgs or []) + messages
            
            formatted_messages = self.format_messages(all_msgs, supports_images)

            if not formatted_messages:
                logger.warning("Formatted messages list is empty after processing. Aborting API call.")
                return None

            input_tokens = self.count_message_tokens(formatted_messages)
            
            current_tools = tools
            if self.api_type == "google" and tools:
                current_tools = self._clean_tools_for_gemini(tools)

            if current_tools:
                tools_tokens = sum(self.count_tokens(str(tool)) for tool in current_tools)
                input_tokens += tools_tokens
            
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            params = {
                "messages": formatted_messages,
                "tools": current_tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                "temperature": temperature if temperature is not None else self.temperature,
                **kwargs,
            }
            
            response = await self._litellm_completion(**params, stream=False)

            if not response.choices or not response.choices[0].message:
                logger.warning("Invalid or empty response from LLM in ask_tool")
                return None

            if response.usage:
                self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
            
            return response.choices[0].message

        except TokenLimitExceeded:
            raise
        except (ValueError, OpenAIError, Exception) as e:
            logger.exception(f"Error in ask_tool: {e}")
            raise
