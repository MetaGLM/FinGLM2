"""This module handles the configuration and data loading for the application."""

import json
import os
import llms

ROOT_DIR = os.getcwd()
with open(ROOT_DIR + "/assets/db_info.json", encoding="utf-8") as file:
    dbs_info = file.read()
with open(ROOT_DIR + "/assets/db_table.json", encoding="utf-8") as file:
    db_table = json.loads(file.read())
with open(ROOT_DIR + "/assets/table_column.json", encoding="utf-8") as file:
    table_column = json.loads(file.read())
with open("../../assets/question.json", "r", encoding="utf-8") as file:
    all_question = json.load(file)


for cols in table_column.values():
    for col in cols:
        # col["desc"] = re.sub(r'(?<=；)[^；]*?与[^；]*?关联', '', col["desc"])
        if col["column"] == "SHKind":
            col["desc"] += (
                "枚举值:资产管理公司,一般企业,投资、咨询公司,风险投资公司,自然人,其他金融产品,信托公司集合信托计划,金融机构—证券公司,保险投资组合,开放式投资基金,企业年金,信托公司单一证券信托,社保基金、社保机构,金融机构—银行,金融机构—期货公司,基金专户理财,国资局,券商集合资产管理计划,基本养老保险基金,金融机构—信托公司,院校—研究院,金融机构—保险公司,公益基金,保险资管产品,财务公司,基金管理公司,金融机构—金融租赁公司"
            )

column_mapping = {}
for db_name, db in dict(db_table).items():
    for table in db["表"]:
        table_name = table["表英文"]
        column_mapping[f"{db_name}.{table_name}"] = {}
        for col in table_column[table_name]:
            column_mapping[f"{db_name}.{table_name}"][col["column"]] = str(col["desc"]).split("；", maxsplit=1)[0]

import_column_names = {
    "InnerCode",
    "CompanyCode",
    "SecuCode",
    "ChiNameAbbr",
    "ChiSpelling",
    "ConceptCode",
    "FirstIndustryCode",
    "SecondIndustryCode",
    "ThirdIndustryCode",
    "FourthIndustryCode",
    "IndustryNum",
    "IndexCode",
    "IndexInnerCode",
    "SecuInnerCode",
    "FirstPublDate",
}

enum_columns = {}
for t_name, table in table_column.items():
    filtered_columns = {col["column"]: col["desc"] for col in table if "具体描述" in col["desc"]}
    if filtered_columns:
        enum_columns[t_name] = filtered_columns

MAX_ITERATE_NUM = 20
MAX_SQL_RESULT_ROWS = 100

START_INDEX = [0, 0]  # 起始下标 [team_index, question_idx]
END_INDEX = [len(all_question) - 1, len(all_question[-1]["team"]) - 1]  # 结束下标 [team_index, question_idx] (包含)
SAVE_FILE_SUBFIX = ""

llm_plus = llms.llm_glm_4_plus
