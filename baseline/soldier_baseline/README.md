# 逐个突破·精准召回——GLM子问题优化与RAG高效检索

***
成绩验证分数: 55.6分

成绩验证时间: 2025年2月7日 9:00 AM

完整运行时长: 8小时2分钟

完整运行Token消耗: GLM-4-Plus 2022万 Tokens

其中: `MODEL_sql = "glm-4-plus"`, `MODEL_rag = "glm-4-plus"`

***


参赛方自测分数: MODEL_sql和MODEL_rag具体含义请看[这里](#1)

* 55.67分 (`MODEL_sql = "glm-4-plus"`, `MODEL_rag = "glm-4-flashx"`)
* 30.73分 (`MODEL_sql = "glm-4-flashx"`, `MODEL_rag = "glm-4-flashx"`)

完整运行时长: 6.5小时左右(数据预处理3.5小时，推理3小时)

完整运行Token消耗: 约2000万 Tokens

## 项目介绍

soldier 像士兵一样绝对听从命令，提供准确的信息查询。部分工具函数参考了bus_baseline，感谢作者。

* 作者： soldier

### 思路介绍

1. GLM直接解决原始问题比较困难，可以通过不断的解决子问题，来解决原始问题
2. RAG召回的越好，效果越好

### 创新点

#### 针对上述1

* 使用prompt分步写出子问题，解决子问题，最后将所有的对话总结，得到最终答案，逻辑如下，由函数`run_conversation_until_complete`
  实现：

```text
1. 输入问题，生成第一个子问题及其回复
2. 通过不断对话得到n个子问题的答案，直到GLM不再提出子问题
3. 将对话总结，得到最终答案
```

#### 针对上述2

* 对原始问题进行表召回时
  （1）通过命名实体识别、LSH（局部敏感哈希 Locality Sensitive
  Hashing）计算表中的值与实体的相似度、GLM判断问题类型，初步确定问题涉及的股票市场（A股、港股、美股），用于缩小召回范围
  （2）再通过GLM召回和问题相关的表
  （3）如果问题和某行业相关，则后续提示词会附带这个行业的信息
  （4）对table schema进行了重新生成，以方便GLM理解，一是加入了数据示例，二是删除了值单一的字段
* 加入了根据数据表结构生成的表之间关系，让GLM对表之间的join有更准确的判断
* 每次回答子问题时，使用GLM从关键信息中筛选出与问题相关的信息作为参考；回答下一个子问题时，与一个子问题相关的信息要删除。所有关键信息如下，后续可以继续增加：

```python
# 下面是对模型来说较难的查询关键词，参考了bus_baseline
examples = {
    "查询美股公司信息": "查询美股公司，要同时查询USStockDB.US_CompanyInfo 和 ConstantDB.US_SecuMain中的信息,不要遗漏",
    "查询港股公司信息": "查询港股公司，要同时查询HKStockDB.HK_StockArchives 和 ConstantDB.HK_SecuMain中的信息,不要遗漏",
    "查询的表名中有DailyQuote": "查询AStockMarketQuotesDB.QT_DailyQuote, USStockDB.US_DailyQuote中的特定公司股票时，不要使用其他字段筛选特定公司，示例sql语句:SELECT * FROM AStockDailyQuote WHERE  InnerCode = 1234;",
    "查询某行业市值": "查询某行业市值,示例sql语句:SELECT TotalMV, NegotiableMV, FreeFloatMV FROM AStockIndustryDB.LC_IndustryValuation WHERE date(TradingDay) = '2020-07-02' AND IndustryName = '风电零部件';",
    "进行加减乘除数学计算": "使用sql进行加减乘除数学计算。",
    "比例/百分比是多少": "查询比例/百分比是多少，要考虑加上百分号后再进行四舍五入。",
    "近一个月最高价": "查询近一个月最高价,你写的sql语句可以优先考虑表中已有字段HighPriceRM  近一月最高价(元)",
    "近一个月最低价": "查询近一月最低价(元),你写的sql语句直接调用已有字段LowPriceRM",
    "查询某行业数量": "查询某行业某年数量 示例sql语句:SELECT count(*) as 风电零部件_2021 FROM AStockIndustryDB.LC_ExgIndustry where ThirdIndustryName like '%风电零部件%' and year(InfoPublDate)=2021 and IfPerformed = 1;",
    "某股票/公司属于哪些行业/概念板块？": "查询某股票/公司属于哪些概念板块？ 示例sql语句:SELECT ConceptCode, ConceptName from AStockIndustryDB.LC_ConceptList WHERE ConceptCode IN (SELECT DISTINCT ConceptCode  FROM AStockIndustryDB.LC_COConcept WHERE InnerCode = 1167);",
    "某行业/概念板块有哪些股票/公司？": "查询某概念板块有哪些股票/公司？ 示例sql语句:SELECT InnerCode FROM AStockIndustryDB.LC_COConcept WHERE ConceptCode = 11100021;",
    """持有无限售流通A股数量""": """特别重要一定注意，查询最新更新XXXX年年度报告，机构持有无限售流通A股数量合计InstitutionsHoldProp最多公司代码，优先使用查询sql语句，SELECT *
                            FROM AStockShareholderDB.LC_StockHoldingSt
                            WHERE date(EndDate) = 'XXXX-12-31'
                              AND UpdateTime = (
                                SELECT MAX(UpdateTime)
                                FROM AStockShareholderDB.LC_StockHoldingSt
                                WHERE date(EndDate) = 'XXXX-12-31'
                              ) order by InstitutionsHoldings desc limit 1 ，XXXX代表问题查询年度，sql语句禁止出现group by InnerCode;

                              查询最新更新XXXX年年度报告,公司机构持有无限售流通A股比例合计InstitutionsHoldProp是多少,优先使用查询sql语句，SELECT InstitutionsHoldProp
                            FROM AStockShareholderDB.LC_StockHoldingSt
                            WHERE date(EndDate) = 'XXXX-12-31'
                              AND UpdateTime = (
                                SELECT MAX(UpdateTime)
                                FROM AStockShareholderDB.LC_StockHoldingSt
                                WHERE date(EndDate) = 'XXXX-12-31'
                              ) order by InstitutionsHoldings desc limit 1 ，XXXX代表问题查询年度，sql语句禁止出现group by InnerCode;""",
    "xxx指标 新高 最多的交易日": """
    xxx指标 新高 最多的交易日 要用AStockMarketQuotesDB.CS_StockPatterns现有字段，例子中IfHighestTVRMThree字段可以根据情况灵活调整
        查询成交量创近一季度新高的证券数量和交易日，示例sql语句:
            SELECT count(*) as num, TradingDay  FROM AStockMarketQuotesDB.CS_StockPatterns where  IfHighestTVRMThree=1 group by TradingDay ORDER BY num DESC limit 1;
        查询某日成交量创近一季度新高的证券，示例sql语句:
            SELECT InnerCode, TradingDay  FROM AStockMarketQuotesDB.CS_StockPatterns where  IfHighestTVRMThree=1 and date(TradingDay) = '2021-12-23';
""",
    "新高": """新高 要用AStockMarketQuotesDB.CS_StockPatterns现有字段
        查询今天是2021年01月01日，创近半年新高的股票有几只。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01';
        判断某日 YY-MM-DD  InnerCode XXXXXX 是否创近一周的新高，查询结果1代表是,IfHighestHPriceRW字段可以根据情况灵活调整  SELECT   InnerCode,TradingDay,IfHighestHPriceRW  FROM AStockMarketQuotesDB.CS_StockPatterns
where  date(TradingDay)='2021-12-20' and InnerCode = '311490'""",
    "成交额": """查询这家公司一周内成交额是多少。示例sql语句:SELECT TurnoverValueRW AS TurnoverValueWan
FROM AStockMarketQuotesDB.QT_StockPerformance
WHERE InnerCode = 1289 AND date(TradingDay) = '2021-06-17';""",
    "半年度报告": """查询XXXX年半年度报告的条件为：year(EndDate) = XXXX and InfoSource='半年度报告'""",

}
```

* 每次回答子问题时，使用问题中的关键词匹配，直接从表中匹配出可能和子问题相关的列描述；回答下一个子问题时，与一个子问题相关的列描述要删除。

## 快速开始

1. 运行以下命令以安装所需依赖：

```shell
pip install -r requirements.txt
```

2. 打开soldier_baseline.ipynb文件。确保所有依赖库已正确安装。依次运行文件中的各个单元格，系统将自动完成问答流程。运行完成后，程序会将结果保存到
   result.json 文件中。如果你只想运行部分问题。可以调整start_idx和end_idx的值。具体位置相见 jupyter notebook最后一个代码块。
3. 本方案均使用官方提供数据

## 关键参数： {#1}

* `MODEL_sql = "glm-4-plus"`: 用于生成SQL语句的模型名称，默认plus
* `MODEL_rag = "glm-4-plus"`:
  用于除生成SQL外，进行召回、改写、修复json等功能的模型名称，默认glm-4-plus，如果token数不够，可以换成glm-4-flashx或glm-4-flash，效果相差不大

## 核心功能详解

### 1. 工具函数

* LSH（局部敏感哈希 Locality Sensitive Hashing）计算相似度
* LLM生成数据：`async_llm_chain_call`异步多线程调用GLM等函数
* 找到并修复GLM回复中的json数据：`find_json`, `fix_json`
* 文件读取和表结构整理函数

### 2. 预处理数据

* 改进题目提供的数据表结构：加入数据样例，删除没用的列
* 根据题目提供的数据表结构生成表之间的关系
* 生成关键词的别名，防止关键词匹配时无法找到相关的关键词，比如董事会秘书，别名董秘
* 用LSH算法将人名和公司名的特征向量存储到本地，方面后续查找

### 3. 推理问答流程

* `run_conversation`：根据问题，召回相关表，并调用`run_conversation_until_complete`生成问题答案
* `run_conversation_until_complete`：分步写出子问题，解决子问题，最后将所有的对话总结，得到最终答案，每个原始问题的对话将保存在
  `dialog`中，示例：

```json
[
  {
    "role": "system",
    "content": "\n请写sql帮我查询问题。\n问题：<山东国瓷功能材料股份有限公司2021年9月23日开盘价是多少？>\n已查询获得的事实：<('预处理程序通过表格：ConstantDB.SecuMain 查询到以下内容：\\n [\\n {\\n  \"InnerCode\": 14953,\\n  \"CompanyCode\": 165647,\\n  \"SecuCode\": \"300285\",\\n  \"ChiName\": \"山东国瓷功能材料股份有限公司\",\\n  \"ChiNameAbbr\": \"国瓷材料\",\\n  \"EngName\": \"Shandong Sinocera Functional Material Co.,Ltd.\",\\n  \"EngNameAbbr\": \"SINOCERA MATERIAL\",\\n  \"SecuAbbr\": \"国瓷材料\",\\n  \"ChiSpelling\": \"GCCL\"\\n }\\n] \\n', ['ConstantDB.SecuMain'])>\n表结构：<[{'数据表名': 'AStockMarketQuotesDB.QT_DailyQuote', '数据表结构': [{'列名': 'InnerCode', '中文描述': '证券内部编码', '数据示例': '28'}, {'列名': 'TradingDay', '中文描述': '交易日', '数据示例': '2019-01-02 12:00:00.000'}, {'列名': 'PrevClosePrice', '中文描述': '昨收盘(元)', '数据示例': '4.31'}, {'列名': 'OpenPrice', '中文描述': '今开盘(元)', '数据示例': '4.35'}, {'列名': 'HighPrice', '中文描述': '最高价(元)', '数据示例': '4.41'}, {'列名': 'LowPrice', '中文描述': '最低价(元)', '数据示例': '4.26'}, {'列名': 'ClosePrice', '中文描述': '收盘价(元)', '数据示例': '4.3'}, {'列名': 'TurnoverVolume', '中文描述': '成交量(股)', '数据示例': '10189201.0', '注释': '当证券类别为指数时，股票指数成交量单位是股，基金指数成交量单位是份，债券指数成交量单位是元。'}, {'列名': 'TurnoverValue', '中文描述': '成交金额(元)', '数据示例': '43979784.24'}, {'列名': 'TurnoverDeals', '中文描述': '成交笔数(笔)', '数据示例': '5451'}, {'列名': 'XGRQ', '中文描述': '修改日期', '数据示例': '2019-01-02 03:18:09.923'}, {'列名': 'JSID', '中文描述': 'JSID', '数据示例': '599757489948'}]}, {'数据表名': 'ConstantDB.SecuMain', '数据表结构': [{'列名': 'InnerCode', '中文描述': '证券内部编码', '数据示例': '4427'}, {'列名': 'CompanyCode', '中文描述': '公司代码', '数据示例': '3710'}, {'列名': 'SecuCode', '中文描述': '证券代码', '数据示例': '002056'}, {'列名': 'ChiName', '中文描述': '中文名称', '数据示例': '横店集团东磁股份有限公司'}, {'列名': 'ChiNameAbbr', '中文描述': '中文名称缩写', '数据示例': '横店东磁'}, {'列名': 'EngName', '中文描述': '英文名称', '数据示例': 'Hengdian Group DMEGC Magnetics Co.,Ltd.'}, {'列名': 'EngNameAbbr', '中文描述': '英文名称缩写', '数据示例': 'DMEGC'}, {'列名': 'SecuAbbr', '中文描述': '证券简称', '数据示例': '横店东磁'}, {'列名': 'ChiSpelling', '中文描述': '拼音证券简称', '数据示例': 'HDDC'}, {'列名': 'ExtendedAbbr', '中文描述': '扩位简称', '数据示例': 'None'}, {'列名': 'ExtendedSpelling', '中文描述': '拼音扩位简称', '数据示例': 'None'}, {'列名': 'SecuMarket', '中文描述': '证券市场', '数据示例': '90', '注释': '证券市场(SecuMarket)与(CT_SystemConst)表中的DM字段关联，令LB = 201 AND DM IN (10,12,13,14,15,16,18,40,49,50,52,54,55,56,65,66,67,68,69,70,71,72,73,75,76,77,78,79,80,81,83,84,85,86,87,88,89,90,93,94,95,96,99,100,101,102,103,104,105,106,107,110,161,162,180,200,202,210,230,240,260,280,310,320,390,400,620,630,631,640,641,650,653,654,655,657,658,659,660,661,662,663,664,666,667,66302,66303,66305)，得到证券市场的具体描述：10-上海期货交易所，12-中国银行间外汇市场，13-大连商品交易所，14-上海黄金交易所，15-郑州商品交易所，16-上海票据交易所，18-北京证券交易所，40-芝加哥商业交易所，49-澳大利亚证券交易所，50-新西兰证券交易所，52-埃及开罗及亚历山大证券交易所，54-阿根廷布宜诺斯艾利斯证券交易所，55-巴西圣保罗证券交易所，56-墨西哥证券交易所，65-印度尼西亚证券交易所，66-泰国证券交易所，67-韩国首尔证券交易所，68-东京证券交易所，69-新加坡证券交易所，70-台湾证券交易所，71-柜台交易市场，72-香港联交所，73-一级市场，75-亚洲其他交易所，76-美国证券交易所，77-美国纳斯达克证券交易所，78-纽约证券交易所，79-美国其他交易市场，80-加拿大多伦多证券交易所，81-三板市场，83-上海证券交易所，84-其他市场，85-伦敦证券交易所，86-法国巴黎证券交易所，87-德国法兰克福证券交易所，88-欧洲其他交易所，89-银行间债券市场，90-深圳证券交易所，93-上海银行间同业拆借市场，94-瑞士证券交易所，95-荷兰阿姆斯特丹证券交易所，96-约翰内斯堡证券交易所，99-东京同业拆借市场，100-美国国债回购市场，101-伦敦银行同业拆借市场，102-香港银行同业拆借市场，103-新加坡银行同业拆借市场，104-中国银行同业拆借市场，105-欧元银行同业拆借市场，106-布鲁塞尔证券交易所，107-雅加达证券交易所，110-以色列特拉维夫证券交易所，161-意大利证券交易所，162-哥本哈根证券交易所，180-挪威奥斯陆证券交易所，200-斯德哥尔摩证券交易所，202-伊斯坦布尔证券交易所，210-印度国家证券交易所，230-奥地利维也纳证券交易所，240-西班牙马德里证券交易所，260-爱尔兰证券交易所，280-菲律宾证券交易所，310-机构间私募产品报价与服务系统，320-俄罗斯莫斯科证券交易所，390-里斯本证券交易所，400-芝加哥期权交易所，620-胡志明市证券交易所，630-沪市代理深市市场，631-沪市代理港交所市场，640-深市代理沪市市场，641-深市代理港交所市场，650-国际外汇市场(晨星)，653-上海环境能源交易所，654-北京绿色交易所，655-天津碳排放权交易中心，657-湖北碳排放权交易中心，658-重庆碳排放权交易中心，659-四川联合环境交易所，660-广州碳排放权交易所，661-海峡股权交易中心，662-深圳排放权交易所，663-欧洲能源交易所，664-全国碳排放权交易，666-布达佩斯证券交易所，667-全国温室气体自愿减排交易市场，66302-韩国ETS，66303-加拿大魁北克Cap-and-Trade(CaT)，66305-美国区域温室气体倡议（RGGI）。'}, {'列名': 'SecuCategory', '中文描述': '证券类别', '数据示例': '1', '注释': '证券类别(SecuCategory)与(CT_SystemConst)表中的DM字段关联，令LB = 1177 AND DM IN (1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,26,27,28,29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,55,79,80,211)，得到证券类别的具体描述：1-A股，2-B股，4-大盘，5-国债回购，6-国债现货，7-金融债券，8-开放式基金，9-可转换债券，10-其他，11-企业债券，12-企业债券回购，13-投资基金，14-央行票据，15-深市代理沪市股票，16-沪市代理深市股票，17-资产支持证券，18-资产证券化产品，19-买断式回购，20-衍生权证，21-股本权证，23-商业银行定期存款，26-收益增长线，27-新质押式回购，28-地方政府债，29-可交换公司债，30-拆借，31-信用风险缓释工具，32-浮息债计息基准利率，33-定期存款凭证，35-大额存款凭证，36-债券借贷，37-存款类机构质押式回购，38-存款类机构信用拆借，39-现货，40-货币对，41-中国存托凭证，42-协议回购，43-三方回购，44-利率互换品种，45-标准利率互换合约，46-报价回购，47-标准化票据，55-优先股，79-深市代理港交所股票，80-沪市代理港交所股票，211-自贸区债。'}, {'列名': 'ListedDate', '中文描述': '上市日期', '数据示例': '2006-08-02 12:00:00.000'}, {'列名': 'ListedSector', '中文描述': '上市板块', '数据示例': '1'}, {'列名': 'ISIN', '中文描述': 'ISIN代码', '数据示例': 'CNE000001N70'}, {'列名': 'XGRQ', '中文描述': '更新时间', '数据示例': '2021-04-03 09:24:16.313'}, {'列名': 'JSID', '中文描述': 'JSID', '数据示例': '670757056809'}]}]>\n表之间的关联信息如下：<['ConstantDB.SecuMain.InnerCode=AStockMarketQuotesDB.QT_DailyQuote.InnerCode']>\n表结构中列名可以引用使用,表结构中数据示例只是参考不能引用。\n我们现在开始查询当前问题，请你分步写出查询sql语句，我把查询结果告诉你，你再告诉我下一步，\n注意如果我返回的结果为空或者错误影响下一步调用，请重新告诉我sql语句。\n写sql时，请告诉我<这是第几步，这步做了什么事情>\n等你全部回答完成，不需要进行下一步调用时，记得提示我：<全部完成，答案如下>,将答案总结以json格式给我，只需要总结当前问题。\n查询技巧:sql查询年度时优先使用year()函数。sql查询语句不需要注释，不然会报错。sql中日期条件格式应参考这样date(TradingDay) = 'YYYY-MM-DD'。尽量利用表格中已有的字段。\n"
  },
  {
    "role": "user",
    "content": "下面开始解决问题：山东国瓷功能材料股份有限公司2021年9月23日开盘价是多少？"
  },
  {
    "role": "assistant",
    "content": "<这是第1步，这步做了什么事情> 根据提供的信息，首先需要确定山东国瓷功能材料股份有限公司的证券内部编码（InnerCode），然后使用这个编码来查询2021年9月23日的开盘价。\n\n```sql\nSELECT InnerCode\nFROM ConstantDB.SecuMain\nWHERE ChiName = '山东国瓷功能材料股份有限公司';\n```\n\n请运行上述SQL语句并返回结果。"
  },
  {
    "role": "user",
    "content": "{\n  \"success\": true,\n  \"data\": [\n    {\n      \"InnerCode\": 14953\n    }\n  ],\n  \"count\": 1\n}"
  },
  {
    "role": "assistant",
    "content": "<这是第2步，这步做了什么事情> 使用第一步查询得到的InnerCode，接下来查询2021年9月23日的开盘价。\n\n```sql\nSELECT OpenPrice\nFROM AStockMarketQuotesDB.QT_DailyQuote\nWHERE InnerCode = 14953 AND TradingDay = '2021-09-23';\n```\n\n请运行上述SQL语句并返回结果。"
  },
  {
    "role": "user",
    "content": "{\n  \"success\": true,\n  \"data\": [\n    {\n      \"OpenPrice\": 41.12\n    }\n  ],\n  \"count\": 1\n}"
  },
  {
    "role": "assistant",
    "content": "<这是第3步，这步做了什么事情> 由于已经成功获取了开盘价，现在不需要进一步查询其他列名。以下是查询结果的总结。\n\n```json\n{\n  \"问题\": \"山东国瓷功能材料股份有限公司2021年9月23日开盘价是多少？\",\n  \"答案\": \"41.12元\"\n}\n```"
  },
  {
    "role": "user",
    "content": "未找到SQL语句，请重新生成sql。或者在提示我：<全部完成，答案如下>后，直接回答问题。"
  },
  {
    "role": "assistant",
    "content": "<这是第1步，这步做了什么事情> 由于需要查询山东国瓷功能材料股份有限公司2021年9月23日的开盘价，首先需要找到该公司的证券内部编码。\n\n```sql\nSELECT InnerCode\nFROM ConstantDB.SecuMain\nWHERE ChiName = '山东国瓷功能材料股份有限公司';\n```\n\n请运行上述SQL语句并返回结果。"
  },
  {
    "role": "user",
    "content": "{\n  \"success\": true,\n  \"data\": [\n    {\n      \"InnerCode\": 14953\n    }\n  ],\n  \"count\": 1\n}"
  },
  {
    "role": "assistant",
    "content": "<这是第2步，这步做了什么事情> 根据第一步查询到的证券内部编码（InnerCode），现在查询2021年9月23日的开盘价。\n\n```sql\nSELECT OpenPrice\nFROM AStockMarketQuotesDB.QT_DailyQuote\nWHERE InnerCode = 14953 AND TradingDay = '2021-09-23';\n```\n\n请运行上述SQL语句并返回结果。"
  },
  {
    "role": "user",
    "content": "{\n  \"success\": true,\n  \"data\": [\n    {\n      \"OpenPrice\": 41.12\n    }\n  ],\n  \"count\": 1\n}"
  },
  {
    "role": "assistant",
    "content": "<这是第2步，这步做了什么事情> 已知山东国瓷功能材料股份有限公司的证券内部编码为14953，并且需要查询2021年9月23日的开盘价，可以直接使用以下SQL语句进行查询。\n\n```sql\nSELECT OpenPrice\nFROM AStockMarketQuotesDB.QT_DailyQuote\nWHERE InnerCode = 14953 AND TradingDay = '2021-09-23';\n```\n\n请运行上述SQL语句并返回结果。"
  }
]
```

# 展望

- [ ] 将分步规划和生成sql语句用2个agent分别实现，一个负责分步规划，一个负责生成sql语句
- [ ] 加入bad case
