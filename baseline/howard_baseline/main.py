import os
import json
import copy
import logging
import time
from dotenv import load_dotenv

os.environ["DEBUG"] = "0"
os.environ["SHOW_LLM_INPUT_MSG"] = "1"

load_dotenv()

from src.log import setup_logger, get_logger
import config
from agents import agent_rewrite_question, agent_extract_company
from workflows import sql_query, check_db_structure
from utils import ajust_org_question


def process_question(question_team: dict, team_idx: int) -> dict:
    """
    Processes a team of questions, extracting facts and generating answers.

    Args:
        question_team (dict): A dictionary containing a list of questions to process.
        team_idx(int): index of team

    Returns:
        dict: The processed question team with answers and usage tokens.
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    facts = []
    qas = []
    sql_query.clear_history_facts()
    for q_idx, question_item in enumerate(question_team["team"]):
        qid: str = question_item["id"].strip()  # 声明qid的类型为str
        question = ajust_org_question(question_item["question"])
        if team_idx == config.START_INDEX[0] and q_idx < config.START_INDEX[1]:
            qas.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": question_item["answer"]},
                ]
            )
            if "facts" in question_item:
                facts = question_item["facts"]
            if "sql_results" in question_item:
                sql_query.history_facts = copy.deepcopy(question_item["sql_results"])
            print(f">>>>> 【SKIP】id: {qid}")
            continue
        if team_idx == config.END_INDEX[0] and q_idx > config.END_INDEX[1]:
            print("----- EXIT -----\n")
            return question_team
        start_time = time.time()
        log_file_path = config.ROOT_DIR + f"/output/{qid}.log"
        open(log_file_path, "w", encoding="utf-8").close()
        setup_logger(
            log_file=log_file_path,
            log_level=logging.DEBUG,
        )
        logger = get_logger()

        print(f">>>>> id: {qid}")
        print(f">>>>> Original Question: {question_item['question']}")
        logger.debug("\n>>>>> Original Question: %s\n", question_item["question"])

        # 获取实体内部代码
        agent_extract_company.clear_history()
        answer, _ = agent_extract_company.answer(
            (
                """提取下面这段文字中的实体（如公司名、股票代码、拼音缩写等），如果识别结果是空，那么就回复No Entities."""
                f'''"{question}"'''
            )
        )
        if answer != "" and answer not in facts:
            facts.append(answer)

        # rewrite question
        agent_rewrite_question.clear_history()
        qas_content = [
            f"Question: {qa['content']}" if qa["role"] == "user" else f"Answer: {qa['content']}" for qa in qas
        ]
        new_question, _ = agent_rewrite_question.answer(
            (
                "历史问答:无。\n"
                if len(qas_content) == 0
                else "下面是顺序的历史问答:\n'''\n" + "\n".join(qas_content) + "\n'''\n"
            )
            + f"现在用户继续提问，请根据已知信息，理解当前这个问题的完整含义，并重写这个问题使得单独拿出来看仍然能够正确理解：{question}"
        )
        print(f">>>>> Rewrited Question: {new_question}")

        # 注入已知事实
        key_facts = "已知事实"
        if len(facts) > 0:
            kv = {key_facts: "\n---\n".join(facts)}
            sql_query.agent_master.add_system_prompt_kv(kv)
            check_db_structure.agent_table_selector.add_system_prompt_kv(kv)
            check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
        else:
            sql_query.agent_master.del_system_prompt_kv(key_facts)
            check_db_structure.agent_table_selector.del_system_prompt_kv(key_facts)
            check_db_structure.agent_column_selector.del_system_prompt_kv(key_facts)
        if debug_mode:
            print(f"\n>>>>> {key_facts}:\n" + "\n---\n".join(facts))
        logger.debug("\n>>>>> %s:\n%s", key_facts, "\n---\n".join(facts))

        # 注入历史对话
        key_qas = "历史对话"
        if len(qas_content) > 0:
            kv = {key_qas: "\n".join(qas_content)}
            sql_query.agent_master.add_system_prompt_kv(kv)
            check_db_structure.agent_table_selector.add_system_prompt_kv(kv)
            check_db_structure.agent_column_selector.add_system_prompt_kv(kv)
        else:
            sql_query.agent_master.del_system_prompt_kv(key_qas)
            check_db_structure.agent_table_selector.del_system_prompt_kv(key_qas)
            check_db_structure.agent_column_selector.del_system_prompt_kv(key_qas)

        check_db_structure.clear_history()
        res = check_db_structure.run(inputs={"messages": [{"role": "user", "content": new_question}]})
        db_info = res["content"]

        sql_query.clear_history()

        res = sql_query.run(
            inputs={
                "messages": [
                    {"role": "assistant", "content": db_info},
                    {"role": "user", "content": new_question},
                ]
            }
        )
        question_item["answer"] = res["content"]
        # Caching
        qas.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": question_item["answer"]},
            ]
        )
        elapsed_time = time.time() - start_time
        question_item["usage_tokens"] = {
            agent_extract_company.name: agent_extract_company.usage_tokens,
            agent_rewrite_question.name: agent_rewrite_question.usage_tokens,
            check_db_structure.name: check_db_structure.usage_tokens,
            sql_query.name: sql_query.usage_tokens,
        }
        minutes, seconds = divmod(elapsed_time, 60)
        question_item["use_time"] = f"{int(minutes)}m {int(seconds)}s"
        question_item["facts"] = copy.deepcopy(facts)
        question_item["rewrited_question"] = new_question
        question_item["sql_results"] = copy.deepcopy(sql_query.history_facts)

        print(f">>>>> Answer: {question_item['answer']}")
        print(f">>>>> Used Time: {int(minutes)}m {int(seconds)}s\n")
        with open(config.ROOT_DIR + f"/assets/question.json", "w", encoding="utf-8") as file:
            json.dump(config.all_question, file, ensure_ascii=False, indent=4)
    print(f"----- Completed Team Index {i} -----\n")
    return question_team


for i in range(config.START_INDEX[0], config.END_INDEX[0] + 1):
    print(f"----- Processing Team Index {i} ... -----\n")
    try:
        process_question(config.all_question[i], i)
    except Exception as exc:
        print(f"\n***** Team Index {i} generated an exception: {exc} *****\n")

total_usage_tokens = {
    agent_extract_company.name: 0,
    agent_rewrite_question.name: 0,
    check_db_structure.name: 0,
    sql_query.name: 0,
}

for q_team in config.all_question:
    for q_item in q_team["team"]:
        if "usage_tokens" in q_item:
            for key in q_item["usage_tokens"]:
                if key in total_usage_tokens:
                    total_usage_tokens[key] += q_item["usage_tokens"][key]

print(json.dumps(total_usage_tokens, ensure_ascii=False, indent=4))

total_tokens = sum(total_usage_tokens.values())
print(f"所有tokens数: {total_tokens}")

for q_team in config.all_question:
    for q_item in q_team["team"]:
        if "usage_tokens" in q_item:
            del q_item["usage_tokens"]
        if "use_time" in q_item:
            del q_item["use_time"]
        if "iterate_num" in q_item:
            del q_item["iterate_num"]
        if "facts" in q_item:
            del q_item["facts"]
        if "rewrited_question" in q_item:
            del q_item["rewrited_question"]
        if "sql_results" in q_item:
            del q_item["sql_results"]

with open(config.ROOT_DIR + f"/output/Eva_Now_result{config.SAVE_FILE_SUBFIX}.json", "w", encoding="utf-8") as f:
    json.dump(config.all_question, f, ensure_ascii=False, indent=4)
