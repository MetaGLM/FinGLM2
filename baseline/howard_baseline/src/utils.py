"""
This module provides utility functions.
"""

import re
import json
from typing import Optional

COLUMN_LIST_MARK = "数据表的字段信息如下"


def generate_markdown_table(data_list, key_title_map):
    """
    根据输入的数据列表和键标题映射生成 Markdown 表格。

    :param data_list: 包含字典的列表，每个字典代表一行数据。
    :param key_title_map: 字典，键为数据字典中的键，值为表格标题。
    :return: 生成的 Markdown 表格字符串。
    """
    # 创建表头
    headers = "| " + " | ".join(key_title_map.values()) + " |\n"
    separators = "| " + " | ".join("---" for _ in key_title_map) + " |\n"
    markdown_table = headers + separators

    # 填充表格内容
    for item in data_list:
        row = "| " + " | ".join(item[key].replace("\n", "\\n") for key in key_title_map) + " |\n"
        markdown_table += row

    return markdown_table


def get_column_list(db_table, table_column, tables: list[str]) -> str:
    """
    tables: list of table names, format is database_name.table_name
    """
    column_lists = []
    for table in tables:
        if "." not in table or table.count(".") != 1:
            raise ValueError(f"发生异常: 表名`{table}`格式不正确，应该为`database_name.table_name`")
        db_name, table_name = table.split(".")
        if db_name not in db_table:
            raise KeyError(f"发生异常: 数据库名`{db_name}`不存在")
        if any(t["表英文"] == table_name for t in db_table[db_name]["表"]):
            column_lists.append(
                {
                    "表名": table,
                    "表字段": table_column[table_name],
                }
            )
    result = f"已取得可用的{COLUMN_LIST_MARK}:\n" + json.dumps(column_lists, ensure_ascii=False) + "\n"
    return result


def extract_last_sql(query_string: str, block_mark: str) -> Optional[str]:
    """
    从给定的字符串中提取最后一组 SQL 语句，并去掉注释。

    :param query_string: 包含 SQL 语句的字符串。
    :param block_mark: SQL 代码块的标记。
    :return: 最后一组 SQL 语句。
    """
    # 使用正则表达式匹配 SQL 语句块
    sql_pattern = re.compile(rf"(?s)```{re.escape(block_mark)}\s+(.*?)\s+```")
    matches = sql_pattern.findall(query_string)
    if matches:
        # 提取最后一个 SQL 代码块
        last_sql_block = matches[-1].strip()
        # 去掉注释但保留分号
        last_sql_block = re.sub(r"--.*(?=\n)|--.*$", "", last_sql_block)
        # 分割 SQL 语句
        sql_statements = [stmt.strip() for stmt in last_sql_block.split(";") if stmt.strip()]
        # 返回最后一个非空 SQL 语句
        return sql_statements[-1] + ";" if sql_statements else None
    return None


def count_total_sql(query_string: str, block_mark: str) -> int:
    """
    从给定的字符串中提取所有 SQL 语句的总数。

    :param query_string: 包含 SQL 语句的字符串。
    :param block_mark: SQL 代码块的标记。
    :return: SQL 语句的总数。
    """
    # 使用正则表达式匹配 SQL 语句块
    sql_pattern = re.compile(rf"(?s)```{re.escape(block_mark)}\s+(.*?)\s+```")
    matches = sql_pattern.findall(query_string)
    total_sql_count = 0
    for sql_block in matches:
        # 去掉注释但保留分号
        sql_block = re.sub(r"--.*(?=\n)|--.*$", "", sql_block)
        # 分割 SQL 语句并计数
        sql_statements = [stmt.strip() for stmt in sql_block.split(";") if stmt.strip()]
        total_sql_count += len(sql_statements)
    return total_sql_count


def extract_last_json(text: str) -> Optional[str]:
    """
    从给定文本中提取最后一个```json和```之间的内容。

    Args:
        text (str): 包含JSON内容的文本。

    Returns:
        Optional[str]: 提取的JSON字符串，如果未找到则返回None。
    """
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


def show(obj):
    """
    打印对象的 JSON 表示。
    """
    if isinstance(obj, dict):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    elif isinstance(obj, list):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    elif isinstance(obj, str):
        if str(obj).startswith(("{", "[")):
            try:
                o = json.loads(str)
                print(json.dumps(o, ensure_ascii=False, indent=2))
            except Exception:
                print(obj)
        else:
            print(obj)
    elif isinstance(obj, (int, float)):
        print(obj)
    else:
        print(obj)
