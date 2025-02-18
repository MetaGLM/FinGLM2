"""
This module provides implementations of agent
and their interactions with various APIs.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List, Dict
from src.llm import LLM, DEBUG_OPTION_PRINT_TOOL_CALL_RESULT
from src.log import get_logger


@dataclass
class AgentConfig:
    """Configuration settings for the Agent class."""

    llm: LLM
    name: str
    role: str
    constraint: Optional[str] = None
    output_format: Optional[str] = None
    knowledge: Optional[str] = None
    tools: Optional[List[Dict]] = None
    funcs: Optional[List[Callable]] = None
    retry_limit: int = 3
    enable_history: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = None
    debug_tool_call_result: bool = True
    system_prompt_kv: Optional[Dict] = field(default_factory=dict)
    pre_process: Optional[Callable[["Agent", dict], None]] = None
    post_process: Optional[Callable[[str], str]] = None
    max_history_num: int = 30


class Agent:
    """Represents an agent that interacts with various APIs using a language model."""

    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.role = config.role
        self.llm = config.llm
        self.constraint = config.constraint
        self.output_format = config.output_format
        self.knowledge = config.knowledge
        self.tools = config.tools
        self.history = []
        self.max_history_num = config.max_history_num
        self.usage_tokens = 0  # 总共使用的token数量
        self.retry_limit = config.retry_limit
        self.enable_history = config.enable_history
        self.options = {}
        if config.temperature is not None:
            self.options["temperature"] = config.temperature
        if config.top_p is not None:
            self.options["top_p"] = config.top_p
        self.stream = config.stream
        self.debug_tool_call_result = config.debug_tool_call_result
        if config.funcs is not None:
            self.funcs = {func.__name__: func for func in config.funcs}
        else:
            self.funcs = None
        if config.system_prompt_kv is not None:
            self.system_prompt_kv = config.system_prompt_kv
        else:
            self.system_prompt_kv = {}
        self.pre_process = config.pre_process
        self.post_process = config.post_process

    def clear_history(self):
        """Clears the agent's conversation history and resets token counts."""
        self.history = []
        self.usage_tokens = 0

    def add_system_prompt_kv(self, kv: dict):
        """Sets the system prompt key-value pairs for the agent."""
        for k, v in kv.items():
            self.system_prompt_kv[k] = v

    def del_system_prompt_kv(self, key: str):
        """Deletes the specified key from the system prompt key-value pairs for the agent."""
        if key in self.system_prompt_kv:
            del self.system_prompt_kv[key]

    def clear_system_prompt_kv(self):
        """
        Clear the agent's additional system prompt settings
        """
        self.system_prompt_kv = {}

    def get_system_prompt(self):
        """Generates and returns the system prompt based on the agent's attributes."""
        system_prompt = f"## 角色描述\n{self.role}"
        if self.constraint is not None:
            system_prompt += f"\n\n## 约束要求\n{self.constraint}"
        if self.output_format is not None:
            system_prompt += f"\n\n## 输出格式\n{self.output_format}"
        if self.knowledge is not None:
            system_prompt += f"\n\n## 知识库\n{self.knowledge}"
        for key, value in self.system_prompt_kv.items():
            system_prompt += f"\n\n## {key}\n{value}"
        return system_prompt

    def chat(self, messages: list[dict]) -> Tuple[str, int]:
        """Attempts to generate a response from the language model, retrying if necessary.
        return:
            - str: assistant's answer
            - int: usage_tokens
        """
        debug_mode = os.getenv("DEBUG", "0") == "1"
        show_llm_input_msg = os.getenv("SHOW_LLM_INPUT_MSG", "0") == "1"
        logger = get_logger()

        if self.pre_process is not None:
            self.pre_process(self, messages)
        usage_tokens = 0
        for attempt in range(self.retry_limit):
            if attempt > 0:
                if debug_mode:
                    print(f"\n重试第 {attempt} 次...\n")
                logger.info("\n重试第 %d 次...\n", attempt)
            response = ""
            try:
                msgs = (
                    messages
                    if attempt == 0
                    else messages
                    + [{"role": "assistant", "content": response}, {"role": "user", "content": "请修正后重试"}]
                )
                if show_llm_input_msg:
                    if debug_mode:
                        print(f"\n\n>>>>> 【{msgs[-1]['role']}】 Said:\n{msgs[-1]['content']}")
                    logger.debug("\n\n>>>>> 【%s】 Said:\n%s", msgs[-1]["role"], msgs[-1]["content"])
                if debug_mode:
                    print(f"\n\n>>>>> Agent【{self.name}】 Said:")
                logger.debug("\n\n>>>>> Agent【%s】 Said:\n", self.name)
                response, token_count, ok = self.llm.generate_response(
                    system=self.get_system_prompt(),
                    messages=msgs,
                    tools=self.tools,
                    funcs=self.funcs,
                    options=self.options,
                    stream=self.stream,
                    debug_options={DEBUG_OPTION_PRINT_TOOL_CALL_RESULT: self.debug_tool_call_result},
                )
                usage_tokens += token_count
                self.usage_tokens += token_count
                if ok and self.post_process is not None:
                    response = self.post_process(response)
            except Exception as e:
                if debug_mode:
                    print(f"\n发生异常：{str(e)}")
                logger.debug("\n发生异常：%s", str(e))
                ok = False
                response += f"\n发生异常：{str(e)}"
            if ok:  # 如果生成成功，退出重试
                break
        else:
            response, token_count = f"发生异常：{response}", 0  # 如果所有尝试都失败，返回默认值
            return response, token_count

        if self.enable_history:
            self.history = messages + [{"role": "assistant", "content": response}]
            if len(self.history) > self.max_history_num:
                half = len(self.history) // 2 + 1
                # 浓缩一半的history
                if debug_mode:
                    print(f"\n\n>>>>> Agent【{self.name}】 Compress History:")
                logger.debug("\n\n>>>>> Agent【%s】 Compress History:\n", self.name)
                try:
                    compressed_msg, token_count, ok = self.llm.generate_response(
                        system="请你把所有历史对话浓缩成一段话，必须保留重要的信息，不要换行，不要有任何markdown格式",
                        messages=self.history[:half],
                        stream=self.stream,
                    )
                    usage_tokens += token_count
                    self.usage_tokens += token_count
                    if ok:
                        self.history = [{"role": "assistant", "content": compressed_msg}] + self.history[half:]
                except Exception as e:
                    if debug_mode:
                        print(f"\n发生异常：{str(e)}")
                    logger.debug("\n发生异常：%s", str(e))
        return response, usage_tokens

    def answer(self, message: str) -> Tuple[str, int]:
        """Generates a response to a user's message using the agent's history.
        return:
            - str: assistant's answer
            - int: usage_tokens
        """
        messages = self.history + [{"role": "user", "content": message}]
        return self.chat(messages=messages)


class AgentTemplate:
    """A template for creating Agent instances with a given configuration."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def create_agent_instance(self) -> Agent:
        """创建一个Agent实例"""
        return Agent(self.config)
