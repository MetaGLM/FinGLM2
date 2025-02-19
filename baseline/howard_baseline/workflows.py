"""This module initializes Workflows."""

import config
from src.workflow import SqlQuery, CheckDbStructure
from utils import execute_sql_query, db_select_post_process, table_select_post_process, foreign_key_hub

sql_query = SqlQuery(
    execute_sql_query=execute_sql_query,
    llm=config.llm_plus,
    max_iterate_num=config.MAX_ITERATE_NUM,
    cache_history_facts=True,
    specific_column_desc=config.enum_columns,
    default_sql_limit=config.MAX_SQL_RESULT_ROWS,
)
sql_query.agent_master.add_system_prompt_kv(
    {
        "EXTEND INSTRUCTION": (
            """- 如果Company和InnerCode都搜不到，那么要考虑股票代码\n"""
            """- CompanyCode跟InnerCode不对应，不能写`CompanyCode`=`InnerCode`，可以通过constantdb.secumain、constantdb.hk_secumain或constantdb.us_secumain换取对方\n"""
            """- 涉及股票价格时：\n"""
            """    - 筛选是否新高，要选择`最高价`字段(HighPrice)，而非收盘价(ClosePrice)，比如月度新高要看月最高价(HighPriceRM)，年度新高要看年最高价(HighPriceRY)，周新高要看周最高价(HighPriceRW)\n"""
            """- ConceptCode是数字，不是字符串\n"""
            """- 在lc_actualcontroller中只有1条记录也代表实控人发生了变更\n"""
            """- 如果用户的前一条提问里提及某实体，那么后续追问虽未明说，但也应该是跟该实体相关\n"""
            """- 注意观察同一个表中的类型字段，结合用户的问题，判断是否要进行类型筛选\n"""
            """- 如果用户提问是希望知道名字，那么要把名字查出来\n"""
            """- 中国的城市的AreaInnerCode是constantdb.lc_areacode里ParentName为'中国'的，你不应该也并不能获取到所有中国的城市代码，所以你需要用联表查询\n"""
            """- 我们的数据库查询是有一个默认的LIMIT的，这是个重要的信息，当你的SQL没有明确LIMIT的时候，你要知道获取到的数据可能不是全部。\n"""
            """- 如果用户提问涉及某个年度的“年度报告”，默认该报告是在次年发布。例如，“2019年年度报告”是在2020年发布的。\n"""
            """- 季度报告通常在下一个季度发布，例如，第一季度的报告会在第二季度发布。\n"""
            """- 如果用户想知道子类概念的名称，你应该去获取astockindustrydb.lc_conceptlist的ConceptName和ConceptCode\n"""
            """- A股公司的基本信息在astockbasicinfodb.lc_stockarchives, 港股的在hkstockdb.hk_stockarchives, 美股的在usstockdb.us_companyinfo\n"""
            """- A股公司的上市基本信息在constantdb.secumain, 港股的在constantdb.hk_secumain, 美股的在constantdb.us_secumain\n"""
            """- 作为筛选条件的名称，请务必分清楚它是公司名、人名还是其他什么名称，避免用错字段\n"""
            """- 但凡筛选条件涉及到字符串匹配的，都采取模糊匹配，增加匹配成功概率\n"""
            """- 比例之间的加减乘除，要务必保证算子是统一单位的，比如3%其实是0.03，0.02其实是2%\n"""
            """- 时间日期字段都需要先做`DATE()`或`YEAR()`格式化再参与SQL的筛选条件，否则就扣你20美元罚款\n"""
            """- 关于概念，可以同时把ConceptName、SubclassName、ClassName查询出来，你就对概念有全面的了解，要记住概念有三个级别，据此理解用户提及的概念分别属于哪个级别\n"""
            """- IndustryCode跟CompanyCode不对应，不能写`IndustryCode`=`CompanyCode`\n"""
            """- 指数内部编码（IndexInnerCode）：与“证券主表（constantdb.secumain）”中的“证券内部编码（InnerCode）”关联\n"""
            """- 证券内部编码（SecuInnerCode）：关联不同主表，查询证券代码、证券简称等基本信息。当0<SecuInnerCode<=1000000时，与“证券主表（constantdb.secuMain）”中的“证券内部编码（InnerCode）”关联；当1000000<SecuInnerCode<=2000000时，与“港股证券主表（constantdb.hk_secumain）”中的“证券内部编码（InnerCode）”关联；当7000000<SecuInnerCode<=10000000时，与“ 美股证券主表（constantdb.us_secumain）”中的“证券内部编码（InnerCode）”关联；\n"""
            """- 指数内部代码（IndexCode）：与“证券主表（constaintdb.secuMain）”中的“证券内部编码（InnerCode）”关联\n"""
            """- 假设A表有InnerCode, B表有ConceptCode和InnerCode，我们需要找出B表里的所有InnerCode，然后用这些InnerCode从A表获取统计数据，那么可以用联表查询 SELECT a FROM A WHERE InnerCode in (SELECT InnerCode FROM B WHERE ConceptCode=b)\n"""
            """- 一个公司可以同时属于多个概念板块，所以如果问及一个公司所属的概念板块，指的是它所属的所有概念板块\n"""
            """- ConceptCode跟InnerCode不对应，不能写`ConceptCode`=`InnerCode`\n"""
            """- 如果用户要求用简称，那你要保证获取到简称(带Abbr标识)，比如constantdb.secumain里中文名称缩写是ChiNameAbbr\n"""
            """- 关于分红的大小比较, 如果派现金额(Dividendsum)没记录，那么可以通过税后实派比例(ActualRatioAfterTax)来比价大小\n"""
            """- 不能使用的关键词`Rank`作为别名，比如`SELECT a as Rank;`\n"""
            """- AreaInnerCode跟CompanyCode不对应，不能写`AreaInnerCode`=`CompanyCode`\n"""
        ),
        "INDUSTRY TERMINOLOGY": (
            """- 概念分支指的是是subclass\n"""
            """- "化工"是2级概念(SubclassName)\n"""
            """- 子类概念的字段是ConceptName和ConceptCode，被纳入到2级概念(SubclassName)或者1级概念(ClassName)下\n"""
            """- 基金管理人指的是负责管理基金的公司，而基金经理则是具体负责基金投资运作的个人\n"""
        ),
    }
)
sql_query.agent_summary.output_format = (
    "- 输出的格式，重点关注日期、小数点几位、数字格式（不要有逗号）\n"
    "    例如:"
    "    - 问题里如果要求(XXXX-XX-XX),日期格式应该类似这种 2025-02-04\n"
    "    - 问题里如果要求(XXXX年XX月XX日),日期格式应该类似这种 2025年2月4日\n"
    "    - 问题里如果要求(保留2位小数),数字格式应该类似这种 12.34\n"
    "    - 问题里如果要求(保留4位小数),数字格式应该类似这种 12.3456\n"
    "    - 比较大的数字不要千位分隔符,正确的格式都应该类似这种 12345678\n"
    "- 输出应该尽可能简短，直接回复答案\n"
    "    例如(假设用户的提问是:是否发生变更？金额多大？):\n"
    "    是否发生变更: 是, 金额: 12.34元\n"
)

check_db_structure = CheckDbStructure(
    dbs_info=config.dbs_info,
    db_table=config.db_table,
    table_column=config.table_column,
    db_selector_llm=config.llm_plus,
    table_selector_llm=config.llm_plus,
    column_selector_llm=config.llm_plus,
    import_column_names=config.import_column_names,
    db_select_post_process=db_select_post_process,
    table_select_post_process=table_select_post_process,
    foreign_key_hub=foreign_key_hub(),
)
check_db_structure.agent_db_selector.add_system_prompt_kv(
    {
        "EXTEND INSTRUCTION": (
            """- 根据当前已知的数据库的介绍，判断所需数据可能存储在哪些表，不要过度臆测某些数据库可能存在什么数据，要看它的介绍里提到包含哪些数据。\n"""
            """- 见证公司的年度股东大会，意思是出席了年度股东大会作见证\n"""
            """- constantdb.us_secumain.DelistingDate、constantdb.hk_secumain.DelistingDate是退市日期，涉及退市的应该考虑它们\n"""
            """- 概念板块在astockindustrydb\n"""
            """- 行政区划在数据库constantdb的lc_areacode表，但凡涉及到要做行政区划筛选的，都需要把这个数据库选上\n"""
        ),
        "INDUSTRY TERMINOLOGY": (
            """- 概念分支指的是是subclass\n"""
            """- "化工"是2级概念(SubclassName)\n"""
            """- SubclassName不是子类概念,子类概念是指ConceptCode和ConceptName\n"""
            """- 基金管理人指的是负责管理基金的公司，而基金经理则是具体负责基金投资运作的个人\n"""
        ),
    }
)
check_db_structure.agent_table_selector.add_system_prompt_kv(
    {
        "EXTEND INSTRUCTION": (
            """- 涉及股票价格时：\n"""
            """    - 创新高判断必须基于`最高价`字段，而非收盘价\n"""
            """- 年度报告的时间条件应该通过astockbasicinfodb.lc_balancesheetall表的InfoPublDate字段来确认\n"""
            """- constantdb.us_secumain.DelistingDate、constantdb.hk_secumain.DelistingDate是退市日期，涉及退市的应该考虑它们\n"""
            """- 行政区划在数据表constantdb.lc_areacode表，但凡涉及到要做行政区划筛选的，都需要把这个数据表选上\n"""
        ),
        "INDUSTRY TERMINOLOGY": (
            """- 概念分支指的是是subclass\n"""
            """- "化工"是2级概念(SubclassName)\n"""
            """- SubclassName不是子类概念,子类概念是指ConceptCode和ConceptName\n"""
            """- 基金管理人指的是负责管理基金的公司，而基金经理则是具体负责基金投资运作的个人\n"""
        ),
    }
)
check_db_structure.agent_column_selector.add_system_prompt_kv(
    {
        "EXTEND INSTRUCTION": (
            """- 涉及股票价格时：\n"""
            """    - 筛选是否新高，要选择`最高价`字段(HighPrice)，而非收盘价(ClosePrice)，比如月度新高要看月最高价(HighPriceRM)，年度新高要看年最高价(HighPriceRY)，周新高要看周最高价(HighPriceRW)\n"""
            """- 年度报告的时间条件应该通过astockbasicinfodb.lc_balancesheetall表的InfoPublDate字段来确认\n"""
            """- 由于不确定公司是A股还是港股还是美股，所以astockbasicinfodb.lc_stockarchives、hkstockdb.hk_stockarchives、usstockdb.us_companyinfo里的同类字段总要同时选上\n"""
            """- 由于不确定公司是A股还是港股还是美股，所以astockmarketquotesdb.qt_dailyquote、hkstockdb.cs_hkstockperformance、usstockdb.us_dailyquote里的同类字段总要同时选上\n"""
            """- 由于不确定公司是A股还是港股还是美股，所以astockmarketquotesdb.qt_stockperformance、hkstockdb.cs_hkstockperformance里的同类字段总要同时选上\n"""
            """- 作为筛选条件的名称，请务必分清楚它是公司名、人名还是其他什么名称，避免用错字段\n"""
            """- 关于概念，可以同时把ConceptName、SubclassName、ClassName查询出来，你就对概念有全面的了解，要记住概念有三个级别，据此理解用户提及的概念分别属于哪个级别\n"""
            """- 指数内部编码（IndexInnerCode）：与“证券主表（constantdb.secumain）”中的“证券内部编码（InnerCode）”关联\n"""
            """- 证券内部编码（SecuInnerCode）：关联不同主表，查询证券代码、证券简称等基本信息。当0<SecuInnerCode<=1000000时，与“证券主表（constantdb.secuMain）”中的“证券内部编码（InnerCode）”关联；当1000000<SecuInnerCode<=2000000时，与“港股证券主表（constantdb.hk_secumain）”中的“证券内部编码（InnerCode）”关联；当7000000<SecuInnerCode<=10000000时，与“ 美股证券主表（constantdb.us_secumain）”中的“证券内部编码（InnerCode）”关联；\n"""
            """- 指数内部代码（IndexCode）：与“证券主表（constaintdb.secuMain）”中的“证券内部编码（InnerCode）”关联\n"""
            """- 如果用户要求用简称，那你要保证获取到简称(带Abbr标识)，比如constantdb.secumain里中文名称缩写是ChiNameAbbr\n"""
            """- 关于分红的大小比较, 如果派现金额(Dividendsum)没记录，那么可以通过税后实派比例(ActualRatioAfterTax)来比价大小，所以尽量让它们同时被选中\n"""
            """- 行政区划在数据表constantdb.lc_areacode表，但凡涉及到要做行政区划筛选的，都需要把这个数据表的字段选上\n"""
        ),
        "INDUSTRY TERMINOLOGY": (
            """- 概念分支指的是是subclass\n"""
            """- "化工"是2级概念(SubclassName)\n"""
            """- SubclassName不是子类概念,子类概念是指ConceptCode和ConceptName\n"""
            """- 基金管理人指的是负责管理基金的公司，而基金经理则是具体负责基金投资运作的个人\n"""
            """- constantdb.us_secumain.DelistingDate、constantdb.hk_secumain.DelistingDate是退市日期，涉及退市的应该考虑它们\n"""
        ),
    }
)
