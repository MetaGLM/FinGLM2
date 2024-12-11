# 使用RAG来完成股票问答任务

## 项目介绍

### 团队介绍

团队成员: [zR](https://github.com/zRzRzRzRzRzRzR)

萌新大学生一枚，第一次参加比赛，多多包涵。

### 思路介绍

本方案使用了RAG方案来完成股票问答任务，主要思路如下：

1. 构建模型问答 PipeLine

模型问答PipeLine是一个很重要的过程，通过拆分提示词工程来实现让模型每次只执行部分任务，你可以在 [这里](prompt.py)
查看我们的提示词工程。

2. 工具函数

工具函数存放在 [utils.py](utils.py) 中，包括了一些常用的函数，如 `get_answer_by_prompt` 等。

## 快速开始

1. 按照`requirements.txt`安装依赖

```shell
pip install -r requirements.txt
```

2. 对数据进行预处理

```shell
python prepare_dataset.py --excel_file ../../assests/data.xlsx --csv_file ../../assests/data.csv
```

3. 执行主任务，完成问答并生成json文件

```shell
python main.py --input ../../assests/question.json --output outputs/answer.json  --model glm-4-flash --api_key ZHIPUAI_API_KEY
```

## 提交结果

提交结果为 `outputs/answer.json` 文件。
