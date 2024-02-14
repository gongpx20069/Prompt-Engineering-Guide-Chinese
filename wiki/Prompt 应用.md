Prompt 应用

在本指南中，我们将介绍一些高级且有趣的方式，我们可以使用 Prompt 工程来执行有用且更高级的任务。

**请注意，本节内容正在大量开发中。**
主题：
- [生成数据](#生成数据)
- [程序辅助语言模型](#程序辅助语言模型-pal)
- [Python 笔记本](#python-笔记本)


---
## 生成数据
LLMs 在生成文本方面具有强大的能力。使用有效的 Prompt 策略可以引导模型产生更好、更一致、更事实的响应。LLMs 在生成数据方面也特别有用，这对于运行各种实验非常有用。例如，我们可以使用它来快速生成情感分类器的样本，如下所示：

*Prompt:*
```
生成 10 个情感分析的示例。示例被分类为正面或负面。生成 2 个负面示例和 8 个正面示例。使用以下格式：
Q: <句子>
A: <情感>
```

*Output:*
```
Q: 我刚刚收到了有史以来最好的消息！
A: Positive

Q: 我们在工作中得到了加薪！
A: Positive

Q: 我为今天所取得的成就感到非常自豪。
A: Positive

Q: 我今天过得非常愉快！
A: Positive

Q: 我真的很期待周末。
A: Positive

Q: 我刚刚收到了有史以来最好的礼物！
A: Positive

Q: 我现在非常高兴。
A: Positive

Q: 我很幸运有这样一个了不起的家庭。
A: Positive

Q: 外面的天气很阴沉。
A: Negative

Q: 我刚刚收到了一些可怕的消息。
A: Negative
```

这非常有用。我们在指南的另一部分中使用这个示例进行不同的测试。

这是另一个示例，展示了如何利用 LLM 进行数据标注/增强任务。

*Prompt:*
```
生成 3 条葡萄酒评论，并标记与口感、风味、香气相关的 token；以 json 文件形式呈现结果，并添加每个术语的坐标以用于 NER 任务
```

*Output:*
```
[
  {
    "review": "这款霞多丽具有丰富的奶油质地，带有香草和黄油橡木的味道。口感顺滑且平衡，收尾带有热带水果的提示。",
    "tokens": [
      {"text": "丰富", "label": "风味", "start": 5, "end": 9},
      {"text": "奶油", "label": "质地", "start": 10, "end": 16},
      {"text": "香草", "label": "风味", "start": 28, "end": 35},
      {"text": "黄油", "label": "风味", "start": 40, "end": 47},
      {"text": "橡木", "label": "香气", "start": 48, "end": 51},
      {"text": "顺滑", "label": "口感", "start": 61, "end": 67},
      {"text": "平衡", "label": "口感", "start": 68, "end": 81},
      {"text": "提示", "label": "风味", "start": 89, "end": 93},
      {"text": "热带水果", "label": "风味", "start": 98, "end": 112},
      {"text": "收尾", "label": "口感", "start": 116, "end": 122}
    ]
  },
  {
    "review": "这款赤霞珠具有大胆的、饱满的风味，带有黑樱桃和黑巧克力的味道。香气丰富且复杂，带有烟草和皮革的提示。",
    "tokens": [
      {"text": "大胆", "label": "风味", "start": 5, "end": 9},
      {"text": "饱满", "label": "质地", "start": 10, "end": 21},
      {"text": "黑樱桃", "label": "风味", "start": 30, "end": 42},
      {"text": "黑巧克力", "label": "风味", "start": 47, "end": 60},
      {"text": "丰富", "label": "香气", "start": 69, "end": 73},
      {"text": "复杂", "label": "香气", "start": 78, "end": 85},
      {"text": "提示", "label": "香气", "start": 90, "end": 96},
      {"text": "烟草", "label": "香气", "start": 101, "end": 108},
      {"text": "皮革", "label": "香气", "start": 113, "end": 120}
    ]
  },
  {
    "review": "这款雷司令具有清脆、清新的口感，带有青苹果和蜂蜜的味道。香气带有花香和水果香，收尾带有柑橘的提示。",
    "tokens": [
      {"text": "清脆", "label": "质地", "start": 5, "end": 10},
      {"text": "清新", "label": "质地", "start": 12, "end": 22},
      {"text": "青苹果", "label": "风味", "start": 31, "end": 42},
    ]
  }
]
```
---

## 程序辅助语言模型（PAL）

[Gao et al., (2022)](https://arxiv.org/abs/2211.10435) 提出了一种使用 LLMs 阅读自然语言问题并生成程序作为中间推理步骤的方法。所谓的程序辅助语言模型（PAL），与链式思维提示不同，它不是使用自由形式的文本获取解决方案，而是将解决步骤转移到程序运行时，如 Python 解释器。

![](../img/pal.png)

让我们看一个使用 LangChain 和 OpenAI GPT-3 的例子。我们有兴趣开发一个简单的应用程序，能够解释所提出的问题，并通过利用 Python 解释器提供答案。

具体来说，我们有兴趣创建一个函数，允许使用 LLM 回答需要日期理解的问题。我们将为 LLM 提供一个包含一些示例的 Prompt，这些示例来自[这里](https://github.com/reasoning-machines/pal/blob/main/pal/prompt/date_understanding_prompt.py)。

这些是我们需要的导入：

```python
import openai
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
```

让我们先配置一些东西：

```python
load_dotenv()

# API 配置
openai.api_key = os.getenv("OPENAI_API_KEY")

# 对于 LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

设置模型实例：

```python
llm = OpenAI(model_name='text-davinci-003', temperature=0)
```

设置 Prompt + 问题：

```python
question = "今天是 2023 年 2 月 27 日。我正好在 25 年前出生。我出生的日期是 MM/DD/YYYY？"

DATE_UNDERSTANDING_PROMPT = """
# Q: 2015 年将在 36 小时后到来。今天一周后的日期是 MM/DD/YYYY？
# 如果 2015 年将在 36 小时后到来，那么今天是 36 小时前。
today = datetime(2015, 1, 1) - relativedelta(hours=36)
# 一周后的今天，
one_week_from_today = today + relativedelta(weeks=1)
# 格式化为 %m/%d/%Y 的答案是
one_week_from_today.strftime('%m/%d/%Y')
# Q: 2019 年的第一天是星期二，今天是 2019 年的第一个星期一。今天的日期是 MM/DD/YYYY？
# 如果 2019 年的第一天是星期二，今天是 2019 年的第一个星期一，那么今天是 6 天后。
today = datetime(2019, 1, 1) + relativedelta(days=6)
# 格式化为 %m/%d/%Y 的答案是
today.strftime('%m/%d/%Y')
# Q: 音乐会原定于 1943 年 6 月 1 日举行，但被推迟了一天到今天。10 天前的日期是 MM/DD/YYYY？
# 如果音乐会原定于 1943 年 6 月 1 日举行，但被推迟了一天到今天，那么今天是一天后。
today = datetime(1943, 6, 1) + relativedelta(days=1)
# 10 天前，
ten_days_ago = today - relativedelta(days=10)
# 格式化为 %m/%d/%Y 的答案是
ten_days_ago.strftime('%m/%d/%Y')
# Q: 今天是 1969 年 4 月 19 日。24 小时后的日期是 MM/DD/YYYY？
# 今天是 1969 年 4 月 19 日。
today = datetime(1969, 4, 19)
# 24 小时后，
later = today + relativedelta(hours=24)
# 格式化为 %m/%d/%Y 的答案是
today.strftime('%m/%d/%Y')
# Q: 简认为今天是 2002 年 3 月 11 日，但今天实际上是 3 月 12 日，晚了 1 天。24 小时后的日期是 MM/DD/YYYY？
# 如果简认为今天是 2002 年 3 月 11 日，但今天实际上是 3 月 12 日，那么今天是 3 月 1 日。
today = datetime(2002, 3, 12)
# 24 小时后，
later = today + relativedelta(hours=24)
# 格式化为 %m/%d/%Y 的答案是
later.strftime('%m/%d/%Y')
# Q: 简出生在 2001 年 2 月的最后一天。今天是她 16 岁生日。昨天的日期是 MM/DD/YYYY？
# 如果简出生在 2001 年 2 月的最后一天并且今天是她 16 岁生日，那么今天是 16 年后。
today = datetime(2001, 2, 28) + relativedelta(years=16)
# 昨天，
yesterday = today - relativedelta(days=1)
# 格式化为 %m/%d/%Y 的答案是
yesterday.strftime('%m/%d/%Y')
# Q: {question}
""".strip() + '\n'
```

```python
llm_out = llm(DATE_UNDERSTANDING_PROMPT.format(question=question))
print(llm_out)
```

```python
exec(llm_out)
print(born)
```

这将输出以下内容：`02/27/1998`

---
## Python 笔记本

|描述|笔记本|
|--|--|
|学习如何结合使用 Python 解释器和语言模型解决任务。|[程序辅助语言模型](../notebooks/pe-pal.ipynb)|

---

更多示例即将推出！

[上一节 (高级 Prompting)](./prompts-advanced-usage.md)

[下一节 (ChatGPT)](./prompts-chatgpt.md)