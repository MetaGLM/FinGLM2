"""
This module defines various agents for processing SQL queries and rewriting questions.
Each agent is configured with specific roles, constraints, and output formats.
"""

from src.agent import Agent, AgentConfig
from utils import extract_company_code
import config

agent_rewrite_question = Agent(
    AgentConfig(
        name="rewrite_question",
        role=("""你的工作是，根据要求和已有信息，重写用户的问题，让问题清晰明确，把必要的前述含义加进去。"""),
        constraint=(
            """- 不改变原意，不要遗漏信息，特别是时间、回答的格式要求，只返回问题。\n"""
            """- 如果有历史对话，那么根据历史对话，将原问题中模糊的实体（公司、文件、时间等）替换为具体的表述。\n"""
            """- 要注意主语在历史对答中存在继承关系，不能改变了，例如："问:A的最大股东是谁？答:B。问:有多少股东？"改写后应该是"A有多少股东？"\n"""
            """- 如果原问题里存在"假设xxx"这种表述，请一定要保留到重写的问题里，因为它代表了突破某种既定的规则限制，设立了新规则，这是重要信息\n"""
            """- 如果原问题里的时间很模糊，那么考虑是否值得是前一个问答里发生的事件的时间\n"""
        ),
        output_format=("""要求只返回重写后的问题，不要有其他任何多余的输出\n"""),
        llm=config.llm_plus,
        stream=False,
    )
)
agent_extract_company = Agent(
    AgentConfig(
        llm=config.llm_plus,
        name="extract_company",
        role="接受用户给的一段文字，提取里面的实体（如公司名、股票代码、拼音缩写等）。",
        output_format=(
            """```json
["实体名_1", "实体名_2", ...]
```
注意，有可能识别结果为空。"""
        ),
        post_process=extract_company_code,
        enable_history=False,
        stream=False,
    )
)
agent_extract_company.add_system_prompt_kv(
    {
        "ENTITY EXAMPLE": (
            "居然之家",
            "ABCD",
        ),
    }
)
