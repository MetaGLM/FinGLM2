# FinGLM

## 仓库介绍

本仓库是关于 [2024金融行业·大模型挑战赛](https://competitions.zhipuai.cn/matchDetail?id=120241202000000003) 的开源代码汇总，
旨在探索大语言模型在法律行业的应用潜力。 本仓库存放了数个比赛团队的竞赛原始代码，均经过整理并开源。

## 项目更新

- **News**: ```2024/12/11```:  本仓库关于BaseLine提交要求和例子提供。

## 初赛 Baseline 提交说明

### 评测环境说明

- Python 3.12.7 / Python 3.10.13
- Ubuntu 22.04 操作系统, 每个参赛选手会分配到16G运行内存。
- **不提供** GPU / NPU，仅保证`UTF-8` 编码格式能正常解码。
- 仅能访问智谱AI API，pip清华源仓库, 比赛数据库, 不提供联网服务。评测环境提供API KEY，但不具备联网功能，**禁止** 流式输出。
  > 如果你提交的代码依赖源码安装的pip包，请在README中写明。主办方将在审核后从github拉取对应pip包主分支。
  >
  > 你所使用的pip包必须是在2024年12月1日之前发布的版本，主办方将有权对你使用的pip包提问其作用。

### 提交规范

- 不允许提交`.idea, .DS_Store` 等无效文件和本地缓存文件。
- 本仓库会提供比赛的参考材料，放置在[assets](assets),
  所有的预处理的数据工作必须基于这些材料进行，不允许提交明文信息参考材料，比如公司股票信息，公司介绍等用于辅助回答答案的内容，无论这些参考材料用于何处。
- 提交代码文件请使用`ruff`格式刷。确保满足`PEP 8`规范。规范文件请参考这里: [规范文件](pyproject.toml)
- 必须提交一个`README.md`用以解释你的思路，确保复现人员能够理解你的代码, 以及根据`requirements.txt`来安装依赖。若运行失败，则提交无效。
- 必须提交一个`jupyter notebook`文件，用以展示你的代码运行结果。确保代码能够正常运行，否则提交无效。
- 所有`API_KEY`必须用环境变量，或者外部传递变量的方式传递。

请不要直接贡献到这个仓库，而是通过比赛链接上传到比赛官方进行评测。不满足提交规范的作品将不被收录。你可以参考 [例子](baseline/sample/README.md)。
- 每个组最多提交三次。和已有的已经开源的 Baseline 方案相似方案将不会被收录 包含思维方式相似，预处理方式相似，工作流相似。因此，请明确在README说明你的创新点。

###

## 开源协议

本代码中无特殊说明或者无注名额外协议的，均使用 [Apache 2.0](LICENSE) 协议。
