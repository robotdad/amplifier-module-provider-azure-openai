"""
Azure OpenAI provider module for Amplifier.
Integrates with Azure OpenAI Service using Responses API.
"""

import asyncio
import logging
import os
import time
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from openai import AsyncOpenAI

# Try to import Azure Identity for managed identity support
try:
    from azure.identity import DefaultAzureCredential
    from azure.identity import ManagedIdentityCredential
    from azure.identity import get_bearer_token_provider

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Azure OpenAI provider."""
    config = config or {}

    # Get Azure endpoint from config or environment
    azure_endpoint = (
        config.get("azure_endpoint")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_BASE_URL")
    )

    if not azure_endpoint:
        logger.warning("No azure_endpoint found for Azure OpenAI provider")
        return

    # Azure Responses API is at /openai/v1/ path
    # Remove trailing slash if present, then add the v1 path
    azure_endpoint = azure_endpoint.rstrip("/")
    base_url = f"{azure_endpoint}/openai/v1/"

    # Get API version (not used in base_url approach, but keep for reference)
    api_version = config.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-01-preview"

    # Get authentication configuration
    api_key = config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")

    # Get managed identity configuration
    use_managed_identity = config.get("use_managed_identity", False)
    if not use_managed_identity:
        env_val = os.environ.get("AZURE_USE_MANAGED_IDENTITY", "").lower()
        use_managed_identity = env_val in ("true", "1", "yes")

    use_default_credential = config.get("use_default_credential", False)
    if not use_default_credential:
        env_val = os.environ.get("AZURE_USE_DEFAULT_CREDENTIAL", "").lower()
        use_default_credential = env_val in ("true", "1", "yes")

    managed_identity_client_id = config.get("managed_identity_client_id") or os.environ.get(
        "AZURE_MANAGED_IDENTITY_CLIENT_ID"
    )

    # Priority-based authentication
    # Priority 1: API Key
    if api_key:
        provider = AzureOpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            config=config,
            coordinator=coordinator,
        )
        logger.info("Using API key authentication for Azure OpenAI")
    # Priority 2: Managed Identity / Default Credential
    elif use_managed_identity or use_default_credential:
        if not AZURE_IDENTITY_AVAILABLE:
            logger.error(
                "Managed identity authentication requires 'azure-identity' package. "
                "Install with: pip install azure-identity"
            )
            return

        try:
            if use_default_credential:
                credential = DefaultAzureCredential()
                logger.info("Using DefaultAzureCredential for Azure OpenAI")
            elif managed_identity_client_id:
                credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
                logger.info(f"Using user-assigned managed identity (client_id: {managed_identity_client_id})")
            else:
                credential = ManagedIdentityCredential()
                logger.info("Using system-assigned managed identity for Azure OpenAI")

            # Get token provider for OpenAI client
            sync_token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

            # Wrap sync token provider in async function for AsyncOpenAI
            async def async_token_provider():
                return sync_token_provider()

            provider = AzureOpenAIProvider(
                base_url=base_url,
                token_provider=async_token_provider,
                config=config,
                coordinator=coordinator,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure credential: {e}")
            return
    else:
        logger.warning("No authentication method configured for Azure OpenAI provider")
        return

    await coordinator.mount("providers", provider, name="azure-openai")
    logger.info(f"Mounted AzureOpenAIProvider (Responses API, endpoint: {base_url}, version: {api_version})")

    return


class AzureOpenAIProvider:
    """Azure OpenAI Service integration using Responses API (matches OpenAI provider)."""

    name = "azure-openai"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        token_provider: Any = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """Initialize Azure OpenAI provider."""
        # Create client using base_url approach (for Responses API)
        if api_key:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            logger.debug("Using API key authentication")
        elif token_provider:
            # Pass token_provider as api_key - SDK will call it for each request
            self.client = AsyncOpenAI(base_url=base_url, api_key=token_provider)
            logger.debug("Using Azure credential authentication")
        else:
            raise ValueError("No authentication method provided")

        self.config = config or {}
        self.coordinator = coordinator

        # Configuration (same as OpenAI provider)
        self.default_model = self.config.get("default_model") or self.config.get("default_deployment", "gpt-4o")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", None)  # None = not sent
        self.reasoning = self.config.get("reasoning", None)  # None = not sent
        self.enable_state = self.config.get("enable_state", False)
        self.priority = self.config.get("priority", 100)

        logger.debug(f"Configured with default_model: {self.default_model}")

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> ProviderResponse:
        """Generate completion using Azure OpenAI Responses API."""

        # 1. Extract system instructions and convert messages to input
        instructions, remaining_messages = self._extract_system_instructions(messages)
        input_text = self._convert_messages_to_input(remaining_messages)

        # 2. Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_text,
        }

        # Add instructions if present
        if instructions:
            params["instructions"] = instructions

        # Add max output tokens
        if max_tokens := kwargs.get("max_tokens", self.max_tokens):
            params["max_output_tokens"] = max_tokens

        # Add temperature
        if temperature := kwargs.get("temperature", self.temperature):
            params["temperature"] = temperature

        # Add reasoning control
        if reasoning := kwargs.get("reasoning", self.reasoning):
            params["reasoning"] = {"effort": reasoning}

        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = self._convert_tools(kwargs["tools"])

        # Add JSON schema if requested
        if json_schema := kwargs.get("json_schema"):
            params["text"] = {"format": {"type": "json_schema", "json_schema": json_schema}}

        # Handle stateful conversations if enabled
        if self.enable_state:
            params["store"] = True
            if previous_id := kwargs.get("previous_response_id"):
                params["previous_response_id"] = previous_id

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "azure-openai",
                    "model": params["model"],
                    "messages": len(remaining_messages),
                    "reasoning": params.get("reasoning") is not None,
                },
            )

        start_time = time.time()
        try:
            # 3. Call Responses API with timeout
            try:
                response = await asyncio.wait_for(self.client.responses.create(**params), timeout=30.0)
                elapsed_ms = int((time.time() - start_time) * 1000)
            except TimeoutError:
                logger.error("Azure OpenAI Responses API timed out after 30s")
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        "llm:response",
                        {
                            "provider": "azure-openai",
                            "model": params["model"],
                            "status": "error",
                            "duration_ms": int((time.time() - start_time) * 1000),
                            "error": "Timeout after 30 seconds",
                        },
                    )
                raise TimeoutError("Azure OpenAI API request timed out after 30 seconds")

            # 4. Parse response output (same as OpenAI)
            content, tool_calls, content_blocks = self._parse_response_output(response.output)

            # Check for reasoning/thinking blocks
            has_reasoning = False
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                for block in content_blocks or []:
                    if isinstance(block, ThinkingContent):
                        has_reasoning = True
                        await self.coordinator.hooks.emit("thinking:final", {"text": block.text})

            # Emit llm:response success event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "azure-openai",
                        "model": params["model"],
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                        "usage": {
                            "input": getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                            "output": getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                        },
                        "has_reasoning": has_reasoning,
                    },
                )

            # 5. Return standardized response
            return ProviderResponse(
                content=content,
                raw=response,
                usage={
                    "input": getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                    "output": getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                    "total": getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0,
                },
                tool_calls=tool_calls if tool_calls else None,
                content_blocks=content_blocks if content_blocks else None,
            )

        except Exception as e:
            logger.error(f"Azure OpenAI Responses API error: {e}")

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "azure-openai",
                        "model": params.get("model", self.default_model),
                        "status": "error",
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "error": str(e),
                    },
                )

            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """Parse tool calls from provider response."""
        return response.tool_calls or []

    def _extract_system_instructions(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system messages as instructions."""
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        instructions = None
        if system_messages:
            instructions = "\n\n".join([m.get("content", "") for m in system_messages])

        return instructions, other_messages

    def _convert_messages_to_input(self, messages: list[dict[str, Any]]) -> str:
        """Convert message array to single input string."""
        formatted = []

        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")

            # Handle tool messages
            if role == "TOOL":
                tool_id = msg.get("tool_call_id", "unknown")
                formatted.append(f"TOOL RESULT [{tool_id}]: {content}")
            elif role == "ASSISTANT" and msg.get("tool_calls"):
                tool_call_desc = ", ".join([tc.get("tool", "") for tc in msg["tool_calls"]])
                if content:
                    formatted.append(f"{role}: {content} [Called tools: {tool_call_desc}]")
                else:
                    formatted.append(f"{role}: [Called tools: {tool_call_desc}]")
            else:
                formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    def _parse_response_output(self, output: list[Any]) -> tuple[str, list[ToolCall], list[Any]]:
        """Parse output blocks into content, tool calls, and content_blocks."""
        content_parts = []
        tool_calls = []
        content_blocks = []

        for block in output:
            # Handle both SDK objects and dictionaries
            if hasattr(block, "type"):
                block_type = block.type

                if block_type == "message":
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                text = getattr(content_item, "text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                            elif hasattr(content_item, "get") and content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_parts.append(block_content)
                        content_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    reasoning_text = getattr(block, "text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))

                elif block_type == "function_call" or block_type == "tool_call":
                    # Azure uses function_call, align to tool_call
                    tool_calls.append(
                        ToolCall(
                            tool=getattr(block, "name", ""),
                            arguments=getattr(block, "input", {}),
                            id=getattr(block, "call_id", "") or getattr(block, "id", ""),
                        )
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=getattr(block, "call_id", "") or getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            arguments=getattr(block, "input", {}),
                            raw=block,
                        )
                    )
            else:
                # Dictionary format
                block_type = block.get("type")

                if block_type == "message":
                    block_content = block.get("content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_parts.append(block_content)
                        content_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    reasoning_text = block.get("text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))

                elif block_type == "function_call" or block_type == "tool_call":
                    tool_calls.append(
                        ToolCall(
                            tool=block.get("name", ""),
                            arguments=block.get("input", {}),
                            id=block.get("call_id", "") or block.get("id", ""),
                        )
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=block.get("call_id", "") or block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                            raw=block,
                        )
                    )

        content = "\n\n".join(content_parts) if content_parts else ""
        return content, tool_calls, content_blocks

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Responses API format."""
        responses_tools = []

        for tool in tools:
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            responses_tools.append(
                {"type": "function", "name": tool.name, "description": tool.description, "parameters": input_schema}
            )

        return responses_tools
