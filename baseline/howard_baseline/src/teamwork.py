"""
This module defines the Teamwork class,
which coordinates teamwork to solve problems using LLM agents and workflows.
"""

import os

# import re
import json
from typing import Tuple, Optional
from src.llm import LLM
from src.agent import Agent, AgentConfig, AgentTemplate
from src.workflow import Workflow
from src.utils import generate_markdown_table, extract_last_json
from src.log import get_logger


class Teamwork:
    """Coordinates teamwork to solve problems using LLM agents and workflows"""

    def __init__(
        self, llm_coordinator: LLM, llm_deliver: LLM, default_context_len: int = 10, name: Optional[str] = None
    ):
        self.name = "Teamwork" if name is None else name
        self.knowledge = []
        self.context = []  # 记录对话历史 list[dict(role, content)]
        self.default_context_len = default_context_len
        self.final_answer_mark = "Final Answer is:"
        self.agent_list_title_map = {"name": "Agent Name", "backstory": "Backstory", "usecase": "Usecase"}
        self.agent_deliver = Agent(
            AgentConfig(
                name=self.name + ".deliver",
                role="你负责根据context，把针对Problem的回复重新组织语言输出。",
                output_format="回答要简洁。",
                llm=llm_deliver,
                post_process=lambda x: f"Final Answer is: {x}",
            )
        )
        self.agent_map = {
            self.agent_deliver.name: self.agent_deliver,
        }
        self.agent_list = [
            {
                "name": self.agent_deliver.name,
                "backstory": "这是一个负责做最终回复的agent。",
                "usecase": (
                    """如果已经能够对Problem给出最后回复或者要结束任务了，就调用这个agent。\n"""
                    """当需要结束任务时，请务必调用这个agent!!!\n"""
                ),
            }
        ]
        self.agent_coordinator = Agent(
            AgentConfig(
                name=self.name + ".coordinator",
                role=(
                    """你是理智且聪明的会议主持，基于待解决的Problem和参与协作Agent List，安排合适的agent发表意见。"""
                    """你总是call tool来唤起agent工作，直至能够针对Problem给出Final Answer。"""
                ),
                output_format=(
                    """如果某个agent需要被唤起，请在以下格式中输出调用：\n\n"""
                    """CALL_AGENT:\n"""
                    """```json\n"""
                    """{\"agent_name\": \"<AgentName>\", \"instruction\": \"<OptionalInstruction>\"}\n"""
                    """```\n"""
                    """\n\n其中，{<AgentName>}是代理人的名字，<OptionalInstruction>是可选的指令。"""
                ),
                constraint=("""- 一次仅能调用一个agent工作\n"""),
                llm=llm_coordinator,
                enable_history=True,
                # temperature = 0.5,
                # top_p = 0.5,
            )
        )

    def register_agent(self, agent: Agent | AgentTemplate | Workflow, name: str, backstory: str, usecase: str):
        """Registers an agent or workflow with the given details.

        Args:
            agent (Agent|Workflow): The agent or workflow to register.
            name (str): The name of the agent.
            backstory (str): The backstory of the agent.
            usecase (str): The use case for the agent.
        """
        if name in self.agent_map:
            raise ValueError(f"agent_name: {name} has been registered")
        if isinstance(agent, AgentTemplate):
            self.agent_map[name] = agent.create_agent_instance()
        else:
            self.agent_map[name] = agent
        self.agent_list.append(
            {
                "name": name,
                "backstory": backstory,
                "usecase": usecase,
            }
        )

    def clear_history(self):
        """Clears the history for all registered agents and the coordinator."""
        for agent in self.agent_map.values():
            agent.clear_history()
        self.agent_coordinator.clear_history()
        self.context = []

    def final_answer(self, answer: str) -> str:
        """Formats and returns the final answer, optionally printing it in debug mode."""
        debug_mode = os.getenv("DEBUG", "0") == "1"
        if debug_mode:
            print(f"\n\n>>>>> Final Answer: {answer}")
        logger = get_logger()
        logger.debug("\n\n>>>>> Final Answer: %s\n", answer)
        return f"{self.final_answer_mark} {answer}"

    def call_agent(self, agent_name: str, *args, instruction: Optional[str] = None, **kwargs) -> str:
        """Calls the specified agent to provide an answer based on the context.

        Args:
            agent_name (str): The name of the agent to call.
            instruction: Optional[str] = None,  # 可选的指令，用于指导代理的工作
            *args: Additional arguments for the agent. No used, just for illustration.
            **kwargs: Additional keyword arguments for the agent. No used, just for illustration.

        Returns:
            str: The answer provided by the agent.
        """
        _ = args
        _ = kwargs
        if agent_name in self.agent_map:
            agent = self.agent_map[agent_name]
            if agent_name == self.agent_deliver.name:
                messages = self.context + [{"role": "user", "content": "请针对Problem给出最终答复"}]
            elif instruction is not None:
                messages = self.context + [{"role": "user", "content": f"{instruction}\n请按指示行动"}]
            else:
                messages = self.context + [{"role": "user", "content": "请按指示行动"}]

            if isinstance(agent, Agent):
                # 处理Agent类型的逻辑
                answer, _ = agent.chat(messages)
            elif isinstance(agent, Workflow):
                # 处理Workflow类型的逻辑
                response = agent.run(inputs={"messages": messages})
                answer = response["content"]
            else:
                raise TypeError(f"未知类型: {type(agent)}")
            return answer
        else:
            return f"unknown agent_name: {agent_name}"

    def extract_args_for_call_agent(self, text: str) -> Optional[dict]:
        """Extracts agent name and instruction from the given text using regex.

        Args:
            text (str): The text containing the agent call information.

        Returns:
            Optional[dict]:
                A dictionary with 'agent_name' and 'instruction' if found, otherwise None.
        """
        if "CALL_AGENT:" in text:
            args_json = extract_last_json(text=text)
            if args_json is not None:
                return json.loads(args_json)
        return None

    def add_system_prompt_kv(self, kv: dict):
        """Adds a key-value pair to the system prompt for all agents and the coordinator.

        Args:
            kv (dict): The key-value pair to add.
        """
        self.agent_coordinator.add_system_prompt_kv(kv)
        for agent in self.agent_map.values():
            agent.add_system_prompt_kv(kv)

    def del_system_prompt_kv(self, key: str):
        """Deletes the specified key from the system prompt key-value pairs for the agent."""
        self.agent_coordinator.del_system_prompt_kv(key)
        for agent in self.agent_map.values():
            agent.del_system_prompt_kv(key)

    def clear_system_prompt_kv(self):
        """Clears all key-value pairs from the system prompt for all agents and the coordinator."""
        self.agent_coordinator.clear_system_prompt_kv()
        for agent in self.agent_map.values():
            agent.clear_system_prompt_kv()

    def solve(self, problem: str, max_iterate_num: int = 10) -> Tuple[str, int]:
        """Solve a problem using the registered agents,
        iterating until a final answer is found or the maximum iterations are reached.

        Args:
            problem (str): The problem to be solve.
            max_iterate_num (int, optional): The maximum number of iterations. Defaults to 10.

        Returns:
            str: The final answer provided by the agents.
            int: iterate_num
        """
        debug_mode = os.getenv("DEBUG", "0") == "1"
        logger = get_logger()
        # start
        if debug_mode:
            print(f"\n\n>>>>> 【Problem】: {problem}")
        logger.debug("\n\n>>>>> 【Problem】: %s\n", problem)

        self.agent_coordinator.add_system_prompt_kv({"Problem": problem})
        self.agent_coordinator.add_system_prompt_kv(
            {
                # "Agent List": generate_markdown_table(
                #     self.agent_list, self.agent_list_title_map,
                # )
                "Agent List": json.dumps(self.agent_list, ensure_ascii=False),
            }
        )
        self.agent_deliver.add_system_prompt_kv({"Problem": problem})

        self.context.append({"role": "user", "content": f"我们开始解决这个problem:\n{problem}"})

        answer, _ = self.agent_coordinator.chat(
            messages=self.context[-1:]
            + [
                {
                    "role": "user",
                    "content": "现在是否已经能够解决Problem了？请你判断下一个要找哪个agent来回答。务必遵循call agent的格式要求。",
                }
            ],
        )
        args = self.extract_args_for_call_agent(answer)
        if args is not None:
            try:
                answer = self.call_agent(**args)
            except Exception as e:
                answer = f"发生异常：{str(e)}"
            self.context.append({"role": "assistant", "content": f"{args['agent_name']} Said:\n{answer}"})

        iterate_num = 1
        while self.final_answer_mark not in answer and iterate_num < max_iterate_num:
            iterate_num += 1
            answer, _ = self.agent_coordinator.chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            self.context[-1]["content"]
                            + "\n\n现在是否已经能够解决Problem了？请你判断下一个要找哪个agent来回答。务必遵循call agent的格式要求。"
                        ),
                    }
                ],
            )
            args = self.extract_args_for_call_agent(answer)
            if args is not None:
                try:
                    answer = self.call_agent(**args)
                except Exception as e:
                    answer = f"发生异常：{str(e)}"
                self.context.append({"role": "assistant", "content": f"{args['agent_name']} Said:\n{answer}"})
        if self.final_answer_mark not in answer:
            messages = self.context + [{"role": "user", "content": "请针对Problem给出最终答复"}]
            answer, _ = self.agent_deliver.chat(messages)
        return answer.split(self.final_answer_mark, 1)[-1].strip(), iterate_num
