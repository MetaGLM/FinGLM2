# Howard-baseline

***
成绩验证分数: 52.3分

成绩验证时间: 2025年2月18日 9:00 AM

完整运行时长: 7小时10分钟

完整运行Token消耗: GLM-4-Plus 1030万 Tokens

***

### 准备环境变量

这个文件存储了所有的Keys和API接口，你需要在这里填写 GLM API Keys 和 比赛平台 API。

```
cp .env-template .env
```

### 准备数据

执行 assets.ipynb 里的代码，会在 assets 目录下生成以下几个文件:

- db_info.json
- db_table.json
- table_column.json

它们是从比赛原数据里提取出来结构化成json的数据，因为后续跑的过程，会按照
`“选择数据库”->“选择数据表”->“选择字段”` 三个步骤逐层提取跟题目相关的数据库信息。 另外，为了节省tokens消耗，借助了LLM进行了一些文字归纳。

### 配置

在 config.py 文件里可以设定一些配置项。

```
MAX_ITERATE_NUM = 20  # 配置SQL Query工作流的最大迭代次数
MAX_SQL_RESULT_ROWS = 100 # 配置智谱SQL查询接口的LIMIT参数

START_INDEX = [0, 0]  # 起始下标 [team_index, question_idx]
END_INDEX = [len(all_question)-1, len(all_question[-1]["team"])-1] # 结束下标 [team_index, question_idx] (包含)
SAVE_FILE_SUBFIX = ""

llm_plus = llms.llm_glm_4_plus # 配置使用的LLM
```

其中 START_INDEX 和 END_INDEX 配置要跑的题目范围。

### 执行命令

```
PYTHONUNBUFFERED=1 python main.py | tee -a output/main.log
```

执行以上命令，就可以开始跑了。


### DEBUG

在 manual.ipynb 里，可以针对某道题手工跑，精调prompt。请注意，这仅作为调试使用。

### 输出

+ 输出目录在 output 目录下，结构为 `ttt----题目编号----题目大题.log` 
+ `all_question.json` 里会缓存一些中间数据。 
+ 运行结束后，得到 `Eva_Now_result.json` 作为最终提交的答案。


### 思路

1. 召回跟题目相关的数据库信息

由于数据库表总共有77张，涉及到的数据库结构信息非常多，全部给到LLM，一来浪费token，二来可能带来信息丟失的问题，
三来也非常没必要。
本方案的思路是，从数据库->数据表->表字段这三层的顺序，逐层去找。但这个思路严重依赖两点：上层的信息归纳准确度、完备度、区分度；LLM的理解能力。

2. 模拟人机交互，使LLM自问自答，最后找到想要的答案。
   这个重点有两个：
    - 保留数据库查询到的结果，继承到后面的LLM的上下文。
    - 每一轮跟LLM的问答里，模拟user的问题，是的LLM可以继续思考下去。总之，就是prompt工程。


### 改进思路

数据库信息召回这个，实践下来，发现本方案的三层召回法，效果并不太理想，需要大量的人工精调prompt。
可以考虑构建向量库进行搜索，并且对外链字段做关联召回。