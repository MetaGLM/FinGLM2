"""
This module provides specific functions for Game Finglm
"""

import os
import re
import jieba
import json
import requests
from src.log import get_logger
from src.agent import Agent
from src.utils import extract_last_sql, extract_last_json
from src.workflow import COLUMN_LIST_MARK
import config


def execute_sql_query(sql: str) -> str:
    """
    Executes an SQL query using the specified API endpoint and returns the result as a string.

    Args:
        sql (str): The SQL query to be executed.

    Returns:
        str: The result of the SQL query execution.
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    sql = sql.replace("\\n", " ")
    url = "https://comm.chatglm.cn/finglm2/api/query"
    access_token = os.getenv("ZHIPU_ACCESS_TOKEN", "")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    logger = get_logger()
    logger.info("\n>>>>> 查询sql:\n%s\n", sql)
    if debug_mode:
        print(f"\n>>>>> 查询ql:\n{sql}")
    try:
        response = requests.post(
            url, headers=headers, json={"sql": sql, "limit": config.MAX_SQL_RESULT_ROWS}, timeout=30
        )
    except requests.exceptions.Timeout as exc:
        logger.info("请求超时，无法执行SQL查询，请优化SQL")
        if debug_mode:
            print("请求超时，无法执行SQL查询，请优化SQL")
        raise RuntimeError("执行SQL查询超时，请优化SQL后重试。") from exc
    result = response.json()
    if "success" in result and result["success"] is True:
        data = json.dumps(result["data"], ensure_ascii=False)
        logger.info("查询结果:\n%s\n", data)
        if debug_mode:
            print(f"查询结果:\n{data}")
        return data
    logger.info("查询失败: %s\n", result["detail"])
    if debug_mode:
        print("查询失败:" + result["detail"])
    if "Commands out of sync" in result["detail"]:
        raise SyntaxError("不能同时执行多组SQL: " + result["detail"])
    raise RuntimeError(result["detail"])


def keep_db_column_info(agent: Agent, messages: dict) -> None:
    """Stores knowledge from messages into the agent."""
    for msg in messages:
        if COLUMN_LIST_MARK in msg["content"]:
            agent.add_system_prompt_kv(kv={"Known Database Structure": msg["content"]})


def extract_and_execute_sql(message: str) -> str:
    """
    Extracts SQL from a message and executes it, returning the result.

    Args:
        message (str): The message containing the SQL query.

    Returns:
        str: The result of the SQL query execution.
    """
    sql = extract_last_sql(
        query_string=message,
        block_mark="sql",
    )
    if sql is None:
        if "SELECT" in message:
            raise RuntimeError("请把sql写到代码块```sql```中")
        else:
            return message
    result = execute_sql_query(sql=sql)
    return f"{message}\n执行SQL:\n{sql}查询结果是:\n{result}"


def get_constant_column_list(table_column: dict) -> list:
    """
    Retrieves a list of basic columns for constant tables based on the provided table column data.

    Args:
        table_column (dict): A dictionary containing table names as keys and their columns as values.

    Returns:
        list: A list of dictionaries, each containing the table name and its corresponding columns that are part of the constant tables.
    """
    constant_tables = {
        "constantdb.secumain": {
            "InnerCode",
            "CompanyCode",
            "SecuCode",
            "ChiName",
            "ChiNameAbbr",
            "EngName",
            "EngNameAbbr",
            "SecuAbbr",
        },
        "constantdb.hk_secumain": {
            "InnerCode",
            "CompanyCode",
            "SecuCode",
            "ChiName",
            "ChiNameAbbr",
            "EngName",
            "EngNameAbbr",
            "SecuAbbr",
            "FormerName",
        },
        "constantdb.us_secumain": {
            "InnerCode",
            "CompanyCode",
            "SecuCode",
            "ChiName",
            "EngName",
            "SecuAbbr",
        },
        "constantdb.ct_systemconst": {"LB", "LBMC", "MS", "DM"},
        "constantdb.lc_areacode": {
            "AreaInnerCode",
            "ParentNode",
            "IfEffected",
            "AreaChiName",
            "ParentName",
            "AreaEngName",
            "AreaEngNameAbbr",
            "FirstLevelCode",
            "SecondLevelCode",
        },
        "astockindustrydb.lc_conceptlist": {
            "ClassCode",
            "ClassName",
            "SubclassCode",
            "SubclassName",
            "ConceptCode",
            "ConceptName",
            "ConceptEngName",
        },
    }
    column_lists = []
    for table, cols in constant_tables.items():
        _, table_name = table.split(".")
        col_list = {
            "表名": table,
            "表字段": [],
        }
        for col in table_column[table_name]:
            if col["column"] in cols:
                col_list["表字段"].append(col)
        column_lists.append(col_list)
    return column_lists


def ajust_org_question(question: str) -> str:
    if "合并报表调整后" in question:
        question = question.replace("合并报表调整后", "合并报表")
    return question


def query_company(name: str) -> str:
    # name = name.replace("公司", "")
    if name == "":
        return "[]"
    sql = f"""SELECT 'constantdb.secumain' AS TableName, InnerCode, CompanyCode,
    ChiName, EngName, SecuCode, ChiNameAbbr, EngNameAbbr, SecuAbbr, ChiSpelling
FROM constantdb.secumain 
WHERE SecuCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR ChiNameAbbr LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR EngNameAbbr LIKE '%{name}%'
   OR SecuAbbr LIKE '%{name}%'
   OR ChiSpelling LIKE '%{name}%'
UNION ALL
SELECT 'constantdb.hk_secumain' AS TableName, InnerCode, CompanyCode,
ChiName, EngName, SecuCode, ChiNameAbbr, EngNameAbbr, SecuAbbr, ChiSpelling
FROM constantdb.hk_secumain 
WHERE SecuCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR ChiNameAbbr LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR EngNameAbbr LIKE '%{name}%'
   OR SecuAbbr LIKE '%{name}%'
   OR FormerName LIKE '%{name}%'
   OR ChiSpelling LIKE '%{name}%'
UNION ALL
SELECT 'constantdb.us_secumain' AS TableName, InnerCode, CompanyCode,
ChiName, EngName, SecuCode, null as ChiNameAbbr, null as EngNameAbbr, SecuAbbr, ChiSpelling
FROM constantdb.us_secumain 
WHERE SecuCode = '{name}'
   OR ChiName LIKE '%{name}%'
   OR EngName LIKE '%{name}%'
   OR SecuAbbr LIKE '%{name}%'
   OR ChiSpelling LIKE '%{name}%';"""
    return execute_sql_query(sql)


def seg_entities(entity: str) -> list[str]:
    stopwords = ["公司", "基金", "管理", "有限", "有限公司"]
    seg_list = list(jieba.cut(entity, cut_all=False))
    filtered_seg_list = [word for word in seg_list if word not in stopwords]
    return filtered_seg_list


def extract_company_code(llm_answer: str) -> str:
    """Extracts company codes from the given LLM answer.

    Args:
        llm_answer (str): The answer from the LLM containing company information.

    Returns:
        str: A formatted string with extracted company codes.
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    results = []
    try:
        names_json = extract_last_json(llm_answer)
        if names_json is not None:
            names = json.loads(names_json)
            if not isinstance(names, list):
                raise ValueError("names should be a list")
            for name in names:
                rows = json.loads(query_company(name))
                if len(rows) > 0:
                    info = f"{name}的关联信息有:[" if len(rows) == 1 else f"{name}关联信息有多组:["
                    for idx, row in enumerate(rows):
                        col_chi = {}
                        if "TableName" in row:
                            col_chi = config.column_mapping[row["TableName"]]
                        for k, v in dict(row).items():
                            if k == "TableName":
                                info += f"所在数据表是{v};"
                                continue
                            if k in col_chi:
                                info += f"{k}({col_chi[k]})是{v};"
                            else:
                                info += f"{k}是{v};"
                        info += "]" if idx == len(rows) - 1 else "],"
                    results.append(info)

    except Exception as e:
        if debug_mode:
            print(f"extract_company_code::Exception：{str(e)}")
        logger.debug("extract_company_code::Exception：%s", str(e))
    return "\n".join(results)


def foreign_key_hub() -> dict:
    return {
        "constantdb.secumain": {"InnerCode", "CompanyCode", "SecuCode", "SecuAbbr", "ChiNameAbbr"},
        "constantdb.hk_secumain": {"InnerCode", "CompanyCode", "SecuCode", "SecuAbbr", "ChiNameAbbr"},
        "constantdb.us_secumain": {"InnerCode", "CompanyCode", "SecuCode", "SecuAbbr"},
    }


def db_select_post_process(dbs: list[str]) -> list[str]:
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()
    required_dbs = {"astockbasicinfodb", "hkstockdb", "usstockdb"}
    present_dbs = set(dbs)
    missing_dbs = required_dbs - present_dbs
    # 确保所有必需的数据库都存在
    if len(missing_dbs) > 0 and len(missing_dbs) != len(required_dbs):
        if debug_mode:
            print("补充选择db: " + json.dumps(list(missing_dbs), ensure_ascii=False))
        logger.debug("补充选择db: %s", json.dumps(list(missing_dbs), ensure_ascii=False))
        dbs.extend(missing_dbs)

    return list(dbs)


def table_select_post_process(tables: list[str]) -> list[str]:
    debug_mode = os.getenv("DEBUG", "0") == "1"
    logger = get_logger()

    required_tables_list = [
        {
            "astockbasicinfodb.lc_stockarchives",
            "hkstockdb.hk_stockarchives",
            "usstockdb.us_companyinfo",
            "constantdb.lc_areacode",
        },
        {"astockmarketquotesdb.qt_dailyquote", "hkstockdb.cs_hkstockperformance", "usstockdb.us_dailyquote"},
        {"astockmarketquotesdb.qt_stockperformance", "hkstockdb.cs_hkstockperformance"},
        {"publicfunddb.mf_fundprodname", "publicfunddb.mf_fundarchives"},
        {"astockmarketquotesdb.lc_suspendresumption", "constantdb.hk_secumain", "constantdb.us_secumain"},
        {"astockmarketquotesdb.qt_dailyquote", "astockmarketquotesdb.cs_stockpatterns"},
        {"astockshareholderdb.lc_sharestru", "astockshareholderdb.lc_mainshlistnew"},
    ]

    for required_tables in required_tables_list:
        present_tables = set(tables)
        missing_tables = required_tables - present_tables
        # 确保所有必需的数据库都存在
        if len(missing_tables) > 0 and len(missing_tables) != len(required_tables):
            if debug_mode:
                print("\n补充选择table: " + json.dumps(list(missing_tables), ensure_ascii=False))
            logger.debug("\n补充选择table: %s", json.dumps(list(missing_tables), ensure_ascii=False))
            tables.extend(missing_tables)
    return tables
