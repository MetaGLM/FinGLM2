"""
This module provides implementations of large language models (LLM)
and their interactions with various APIs.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import os
import re
import json
from ollama import Client
from zhipuai import ZhipuAI
from openai import OpenAI
from src.log import get_logger

CHAT_OPTION_TEMPERATURE = "temperature"
CHAT_OPTION_TOP_K = "top_k"
CHAT_OPTION_MAX_TOKENS = "max_tokens"

DEBUG_OPTION_PRINT_TOOL_CALL_RESULT = "print_tool_call_result"


class LLM(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def generate_response(
        self,
        system: str,
        messages: list,
        tools: Optional[list[dict]] = None,
        funcs: Optional[dict[str, Callable]] = None,
        options: Optional[dict] = None,
        stream: Optional[bool] = None,
        debug_options: Optional[dict] = None,
        tool_choice: Optional[bool] = None,
    ) -> tuple[str, int, bool]:
        """生成响应的方法，所有LLM都需要实现
        tool_choice: 目前只有openai支持
            - None 代表 auto,由llm自行判断是否要调用tool
            - False 代表none，表示不调用tool
            - True 代表required, 表示必须调用tool
        返回的tuple包含两个元素：
        - str: LLM的回答
        - int: 输入输出总共的token数量
        - bool: 是否成功生成回答

        参数:
        - messages: 一个数组，每个元素是一个dict，包含role和content，例如：
          [
              {"role": "user", "content": "你好"},
              {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
          ]
        其中role支持user、assistant。
        """


class OllamaLLM(LLM):
    """Concrete implementation of LLM using the Ollama API."""

    def __init__(self, host: str, model: str, post_process: Optional[Callable[[str], str]] = None):
        self.host = host
        self.model = model
        self.post_process = post_process
        # 初始化其他必要的参数
        self.client = Client(host)

    def generate_response(
        self,
        system: str,
        messages: list,
        tools: Optional[list[dict]] = None,
        funcs: Optional[dict[str, Callable]] = None,
        options: Optional[dict] = None,
        stream: Optional[bool] = None,
        debug_options: Optional[dict] = None,
        tool_choice: Optional[bool] = None,  # no use yet
    ) -> tuple[str, int, bool]:
        debug_mode = os.getenv("DEBUG", "0") == "1"
        logger = get_logger()
        if options is None:
            options = {}
        if debug_options is None:
            debug_options = {}
        if CHAT_OPTION_MAX_TOKENS in options:
            options["num_ctx"] = options[CHAT_OPTION_MAX_TOKENS]
            options.pop(CHAT_OPTION_MAX_TOKENS)
        options.setdefault("num_ctx", 5120)
        if stream is None:
            stream = True
        # fortest
        # print(system)
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            options=options,
            tools=tools,
            stream=stream,
        )
        content = ""
        tool_calls = []
        token_count = 0
        if stream:
            for piece in response:
                if piece.message.content is not None:
                    content += piece.message.content
                    if debug_mode:
                        print(piece.message.content, end="")
                    logger.debug("%s", piece.message.content)
                if piece.message.tool_calls is not None:
                    tool_calls.extend(piece.message.tool_calls)
                if piece.prompt_eval_count is not None:
                    token_count += piece.prompt_eval_count
                if piece.eval_count is not None:
                    token_count += piece.eval_count
            if debug_mode and content != "":
                print()  # 打印换行以便于调试输出的可读性
            if content != "":
                logger.debug("\n")
        else:
            if response.message.content is not None:
                content = response.message.content
            if response.message.tool_calls is not None:
                tool_calls = response.message.tool_calls
            if response.prompt_eval_count is not None:
                token_count = response.prompt_eval_count
            if response.eval_count is not None:
                token_count += response.eval_count
            if debug_mode and content != "":
                print(content)
            if content != "":
                logger.debug("%s\n", content)
        ok = True
        for tool_call in tool_calls:
            function_call = tool_call["function"]
            function_name = function_call["name"]
            arguments = function_call["arguments"]
            for key, value in arguments.items():
                if isinstance(value, str):
                    try:
                        arguments[key] = json.loads(value)
                    except json.JSONDecodeError:
                        arguments[key] = value  # 保留原值
            if debug_mode:
                print(f"调用函数 {function_name}({arguments})")
            logger.debug("调用函数 %s(%s)\n", function_name, arguments)
            function = (
                (funcs.get(function_name) if funcs is not None else None)
                or globals().get(function_name)
                or locals().get(function_name)
            )
            if function:
                try:
                    content += "\n调用结果:\n" + function(**arguments)
                except Exception as e:
                    if debug_mode:
                        print(f"\n调用结果:\n执行函数{function_name}时发生错误: {str(e)}")
                    logger.debug("\n调用结果:\n执行函数%s时发生错误: %s", function_name, str(e))
                    content += f"\n调用结果:\n执行函数{function_name}时发生错误: {str(e)}"
                    ok = False
            else:
                content += f"\n调用结果:\n未找到名为 {function_name} 的函数, context: {tool_call}"
                ok = False
            if debug_mode and debug_options.get(DEBUG_OPTION_PRINT_TOOL_CALL_RESULT, True):
                print(content)
            if debug_options.get(DEBUG_OPTION_PRINT_TOOL_CALL_RESULT, True):
                logger.debug("%s\n", content)
            if not ok:
                break
        if ok and self.post_process is not None:
            content = self.post_process(content)
        return content.strip(), token_count, ok


class ZhipuLLM(LLM):
    """Concrete implementation of LLM using the ZhipuAI API."""

    def __init__(self, api_key: str, model: str, post_process: Optional[Callable[[str], str]] = None):
        self.api_key = api_key
        self.model = model
        self.post_process = post_process
        self.client = ZhipuAI(api_key=api_key)

    def generate_response(
        self,
        system: str,
        messages: list,
        tools: Optional[list[dict]] = None,
        funcs: Optional[dict[str, Callable]] = None,
        options: Optional[dict] = None,
        stream: Optional[bool] = None,
        debug_options: Optional[dict] = None,
        tool_choice: Optional[bool] = None,  # no use yet
    ) -> tuple[str, int, bool]:
        debug_mode = os.getenv("DEBUG", "0") == "1"
        logger = get_logger()
        if options is None:
            options = {}
        if debug_options is None:
            debug_options = {}
        if stream is None:
            stream = True
        # fortest
        # show(system)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            top_p=options.get(CHAT_OPTION_TOP_K, 0.5),
            temperature=options.get(CHAT_OPTION_TEMPERATURE, 0.5),
            max_tokens=options.get(CHAT_OPTION_MAX_TOKENS, None),
            stream=stream,
            tools=tools,
        )
        content = ""
        tool_calls = []
        token_count = 0
        if stream:
            for piece in response:
                if len(piece.choices) > 0:
                    if piece.choices[0].delta.content is not None:
                        content += piece.choices[0].delta.content
                        if debug_mode:
                            print(piece.choices[0].delta.content, end="")
                        logger.debug("%s", piece.choices[0].delta.content)
                    if piece.choices[0].delta.tool_calls is not None:
                        tool_calls.extend(piece.choices[0].delta.tool_calls)
                if piece.usage is not None:
                    token_count += piece.usage.total_tokens
            if debug_mode and content != "":
                print()  # 打印换行以便于调试输出的可读性
            if content != "":
                logger.debug("\n")
        else:
            if response.choices[0].message.content is not None:
                content = response.choices[0].message.content
            if response.choices[0].message.tool_calls is not None:
                tool_calls = response.choices[0].message.tool_calls
            if response.usage is not None:
                token_count = response.usage.total_tokens
            if debug_mode and content != "":
                print(content)
            if content != "":
                logger.debug("%s\n", content)
        ok = True
        for tool_call in tool_calls:
            function_call = tool_call.function
            function_name = function_call.name
            arguments = json.loads(function_call.arguments)
            if debug_mode:
                print(f"调用函数 {function_name}({arguments})")
            logger.debug("调用函数 %s(%s)\n", function_name, arguments)
            function = (
                (funcs.get(function_name) if funcs is not None else None)
                or globals().get(function_name)
                or locals().get(function_name)
            )
            if function:
                try:
                    content += "\n调用结果:\n" + function(**arguments)
                except Exception as e:
                    if debug_mode:
                        print(f"\n调用结果:\n执行函数{function_name}时发生错误: {str(e)}")
                    logger.debug("\n调用结果:\n执行函数%s时发生错误: %s", function_name, str(e))
                    content += f"\n调用结果:\n执行函数{function_name}时发生错误: {str(e)}"
                    ok = False
            else:
                content += f"\n调用结果:\n未找到名为 {function_name} 的函数, context: {tool_call}"
                ok = False
            if debug_mode and debug_options.get(DEBUG_OPTION_PRINT_TOOL_CALL_RESULT, True):
                print(content)
            if debug_options.get(DEBUG_OPTION_PRINT_TOOL_CALL_RESULT, True):
                logger.debug("%s\n", content)
            if not ok:
                break
        if ok and self.post_process is not None:
            content = self.post_process(content)
        return content.strip(), token_count, ok


def extract_answer_from_r1(text) -> str:
    """
    Removes content enclosed by <think> and </think> tags from the given text.

    Parameters:
    text (str): The input text containing <think> tags.

    Returns:
    str: The text with <think> tags and their content removed.
    """
    # 删除 <think> 和 </think> 标签包围的内容
    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text_without_think


class OpenAILLM(LLM):
    """Concrete implementation of LLM using the OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        post_process: Optional[Callable[[str], str]] = None,
        base_url: Optional[str] = None,
        default_stream: Optional[bool] = False,
    ):
        self.api_key = api_key
        self.model = model
        self.post_process = post_process
        # 初始化其他必要的参数
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        if self.model.startswith("o"):
            self.system_role = "developer"
        else:
            self.system_role = "system"
        self.default_stream = default_stream

    def generate_response(
        self,
        system: str,
        messages: list,
        tools: Optional[list[dict]] = None,
        funcs: Optional[dict[str, Callable]] = None,
        options: Optional[dict] = None,
        stream: Optional[bool] = None,
        debug_options: Optional[dict] = None,
        tool_choice: Optional[bool] = None,  # "none", "auto", "required"
    ) -> tuple[str, int, bool]:
        debug_mode = os.getenv("DEBUG", "0") == "1"
        logger = get_logger()
        if options is None:
            options = {}
        if debug_options is None:
            debug_options = {}
        if stream is None:
            stream = self.default_stream
        tool_choice_str = "auto" if tool_choice is None else "required" if tool_choice is True else "none"
        # fortest
        # print(system)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": self.system_role, "content": system}] + messages,
            temperature=options.get(CHAT_OPTION_TEMPERATURE, 0.5),
            max_tokens=options.get(CHAT_OPTION_MAX_TOKENS, 5120),
            stream=stream,
            tools=tools,
            tool_choice=None if tools is None else tool_choice_str,
        )
        # fortest
        # print(response)

        content = ""
        tool_calls = []
        token_count = 0
        if stream:
            for piece in response:
                if len(piece.choices) > 0:
                    if piece.choices[0].delta.content is not None:
                        content += piece.choices[0].delta.content
                        if debug_mode:
                            print(piece.choices[0].delta.content, end="")
                        logger.debug("%s", piece.choices[0].delta.content)
                    if piece.choices[0].delta.tool_calls is not None:
                        tool_calls.extend(piece.choices[0].delta.tool_calls)
                if piece.usage is not None:
                    token_count += piece.usage.total_tokens
            if debug_mode and content != "":
                print()  # 打印换行以便于调试输出的可读性
            if content != "":
                logger.debug("\n")
        else:
            if response.choices[0].message.content is not None:
                content = response.choices[0].message.content
            if response.choices[0].message.tool_calls is not None:
                tool_calls = response.choices[0].message.tool_calls
            if response.usage is not None:
                token_count = response.usage.total_tokens
            if debug_mode and content != "":
                print(content)
            if content != "":
                logger.debug("%s\n", content)
        ok = True
        for tool_call in tool_calls:
            function_call = tool_call.function
            function_name = function_call.name
            arguments = json.loads(function_call.arguments)
            if debug_mode:
                print(f"调用函数 {function_name}({arguments})")
            logger.debug("调用函数 %s(%s)\n", function_name, arguments)
            function = (
                (funcs.get(function_name) if funcs is not None else None)
                or globals().get(function_name)
                or locals().get(function_name)
            )
            if function:
                try:
                    content += "\n调用结果:\n" + function(**arguments)
                except Exception as e:
                    if debug_mode:
                        print(f"\n调用结果:\n执行函数{function_name}时发生错误: {str(e)}")
                    logger.debug("\n调用结果:\n执行函数%s时发生错误: %s", function_name, str(e))
                    content += f"\n调用结果:\n执行函数{function_name}时发生错误: {str(e)}"
                    ok = False
            else:
                content += f"\n调用结果:\n未找到名为 {function_name} 的函数, context: {tool_call}"
                ok = False
            if debug_mode and debug_options.get(DEBUG_OPTION_PRINT_TOOL_CALL_RESULT, True):
                print(content)
            if debug_options.get(DEBUG_OPTION_PRINT_TOOL_CALL_RESULT, True):
                logger.debug("%s\n", content)
            if not ok:
                break
        if ok and self.post_process is not None:
            content = self.post_process(content)
        return content.strip(), token_count, ok
