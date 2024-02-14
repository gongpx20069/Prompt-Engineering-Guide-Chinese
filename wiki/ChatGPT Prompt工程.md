# ChatGPT Prompt 工程

在本节中，我们将介绍 ChatGPT 的最新 Prompt 工程技术，包括技巧、应用、限制、论文和额外的阅读材料。

**请注意，本节内容正在积极开发中。**

主题：
- [ChatGPT 简介](#chatgpt-简介)
- [回顾对话任务](#回顾对话任务)
- [与 ChatGPT 的对话](#与-chatgpt-的对话)
- [Python 笔记本](#python-笔记本)

---
## ChatGPT 简介

ChatGPT 是 [OpenAI 训练的](https://openai.com/blog/chatgpt) 新模型，可以进行会话式交互。该模型经过训练，可以遵循 Prompt 中的指令，在对话上下文中提供适当的回应。ChatGPT 可以帮助回答问题、提供建议、创作歌词、生成代码等等。

ChatGPT 使用人类反馈的强化学习（RLHF）进行训练。虽然这个模型比以前的 GPT 迭代版本更有能力（也经过训练以减少有害和不真实的输出），但它仍然有限制。让我们通过具体的例子来了解一些能力和限制。

你可以在 [这里](https://chat.openai.com) 使用 ChatGPT 的研究预览版，但在下面的示例中，我们将使用 OpenAI Playground 的 `Chat` 模式。

---
## 回顾对话任务

在[之前的指南](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-basic-usage.md#conversation)中，我们讨论了一些关于对话能力和角色 Prompt 的内容。我们介绍了如何指导 LLM 以特定风格、特定意图、行为和身份进行对话。

让我们回顾之前的基础示例，我们创建了一个能够对问题生成更多技术性和科学性回应的对话系统。

*Prompt:*
```
以下是与一个 AI 研究助理的对话。助理的语气是技术性和科学性的。

人类：你好，你是谁？
AI：问候！我是一个 AI 研究助理。今天我能帮到你什么？
人类：你能告诉我黑洞的形成吗？
AI：
```

从上面的例子中，你可以看到两个重要组成部分：
- **意图** 或解释聊天机器人是什么
- **身份** 指导聊天机器人将使用哪种风格或语气进行回应

上面的简单示例在使用 `text-davinci-003` 的文本完成 API 中效果很好。最近，OpenAI [宣布了 ChatGPT API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)，这是一个更强大、更便宜的模型，名为 `gpt-3.5-turbo`，专为这种功能（聊天完成）构建。OpenAI 甚至推荐这是他们最好的模型，即使是非聊天用例也是如此。使用 ChatGPT API 的其他好处包括显著的成本降低（90%）和效率提升。

像 Snap Inc. 和 Instacart 这样的大公司已经在其产品中整合了由 ChatGPT 提供动力的对话功能，范围从个性化推荐到开放式购物目标。

---
## 与 ChatGPT 的对话

### 多轮对话

为了展示 ChatGPT 的能力，我们将使用上面的聊天助手示例并讨论结果。与 `text-davinci-003` 相比，为 ChatGPT 提供动力的 `gpt-3.5-turbo` 模型使用聊天格式作为输入。模型期望输入一系列消息，并使用这些消息生成响应。

*Input:*
```
SYSTEM: 你是一个 AI 研究助理。你使用的语气是技术性和科学性的。
USER: 你好，你是谁？
ASSISTANT: 问候！我是一个 AI 研究助理。今天我能帮到你什么？
USER: 你能告诉我黑洞的形成吗？
```

*Output:*
```
ASSISTANT: 当然可以！黑洞是由于大质量恒星耗尽燃料并在自身重力作用下坍塌而形成的。这种坍塌使恒星变得极其密集，具有如此强大的引力场，以至于连光都无法逃脱。这个无法返回的点被称为事件视界。一旦恒星穿过事件视界，它就被认为是黑洞...
```

请注意，在上面的例子中，我简化了输入和输出，但 ChatGPT 聊天完成 API 要求消息必须是特定格式。我在下面添加了一个快照，展示了使用 OpenAI Playground 的 `Chat Mode` 时这个示例的样子：

![](../img/chatgpt-1.png)

我们示例的更正式 API 调用可能看起来像下面的例子：

```python
import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "你是一个 AI 研究助理。你使用的语气是技术性和科学性的。"},
        {"role": "user", "content": "你好，你是谁？"},
        {"role": "assistant", "content": "问候！我是一个 AI 研究助理。今天我能帮到你什么？"},
        {"role": "user", "content": "你能告诉我黑洞的形成吗？"}
    ]
)
```
未来与 ChatGPT 交互的开发者预计将通过 [Chat Markup Language](https://github.com/openai/openai-python/blob/main/chatml.md)（简称 ChatML）进行。

### 单轮任务

聊天格式支持多轮对话，但它也支持类似于我们使用 `text-davinci-003` 执行的单轮任务。这意味着我们可以使用 ChatGPT 执行与我们为原始 GPT 模型演示的类似任务。例如，让我们尝试使用 ChatGPT 执行以下问答任务：

*Input:*
```
USER: 根据下面的上下文回答问题。保持答案简短且精确。如果不确定答案，请回复“不确定答案”。

上下文：Teplizumab 的起源可以追溯到一个名为 Ortho Pharmaceutical 的新泽西药物公司。在那里，科学家们生成了一种早期版本的抗体，称为 OKT3。最初源自老鼠，这种分子能够与 T 细胞表面结合，并限制它们的细胞杀伤潜力。1986 年，它被批准用于帮助预防肾移植后的器官排斥，成为首个获准用于人类的治疗性抗体。

问题：OKT3 最初是从什么来源的？

答案：
```

*Output:*
```
ASSISTANT: 老鼠。
```

请记住，我添加了 `USER` 和 `ASSISTANT` 标签，以更好地演示如何使用 ChatGPT 执行任务。这是使用 Playground 的示例：

![](../img/chatgpt-classic.png)

更正式的 API 调用（我只包含了请求的消息部分）：

```python
CONTENT = """根据下面的上下文回答问题。保持答案简短且精确。如果不确定答案，请回复\"不确定答案\"。

上下文：Teplizumab 的起源可以追溯到一个名为 Ortho Pharmaceutical 的新泽西药物公司。在那里，科学家们生成了一种早期版本的抗体，称为 OKT3。最初源自老鼠，这种分子能够与 T 细胞表面结合，并限制它们的细胞杀伤潜力。1986 年，它被批准用于帮助预防肾移植后的器官排斥，成为首个获准用于人类的治疗性抗体。

问题：OKT3 最初是从什么来源的？

答案：
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": CONTENT},
    ],
    temperature=0,
)
```

### 指导 Chat 模型

根据官方 OpenAI 文档，`gpt-3.5-turbo` 模型的快照也将被提供。例如，我们可以访问 3 月 1 日的快照 `gpt-3.5-turbo-0301`。这允许开发者选择特定版本的模型。这也意味着指导模型的最佳实践可能会随版本而变化。

目前对于 `gpt-3.5-turbo-0301` 的建议是在 `user` 消息中添加指令，而不是可用的 `system` 消息。

---
## Python 笔记本

|描述|笔记本|
|--|--|
|了解如何使用 `openai` 库对 ChatGPT API 进行调用。|[ChatGPT API 简介](../notebooks/pe-chatgpt-intro.ipynb)|
|了解如何使用 `LangChain` 库使用 ChatGPT 功能。|[ChatGPT API 与 LangChain](../notebooks/pe-chatgpt-langchain.ipynb)|

---
## 参考资料

- [通过学生眼中看 ChatGPT：TikTok 数据分析](https://arxiv.org/abs/2303.05349)（2023 年 3 月）
- [使用对话语言模型和 Prompt 工程从研究论文中提取准确的材料数据 - ChatGPT 示例](https://arxiv.org/abs/2303.05352)（2023 年 3 月）
- [ChatGPT 即将到来：大型语言模型是否能满足我们对智能交通的需要？](https://arxiv.org/abs/2303.05382)（2023 年 3 月）
- [制作计算机律师](https://arxiv.org/abs/2303.05383)（2023 年 3 月）
- [LLMs 的临床文本挖掘合成数据生成有帮助吗？](https://arxiv.org/abs/2303.04360)（2023 年 3 月）
- [MenuCraft：使用大型语言模型的交互式菜单系统设计](https://arxiv.org/abs/2303.04496)（2023 年 3 月）
- [AI 生成内容（AIGC）的全面调查：从 GAN 到 ChatGPT 的生成 AI 历史](https://arxiv.org/abs/2303.04226)（2023 年 3 月）
- [探索 ChatGPT 在事件提取中的可行性](https://arxiv.org/abs/2303.03836)
- [ChatGPT：手动注释的终结者？自动体裁识别用例](https://arxiv.org/abs/2303.03953)（2023 年 3 月）
- [ChatGPT 是一个好的 NLG 评估者吗？初步研究](https://arxiv.org/abs/2303.04048)（2023 年 3 月）
- [情感计算会从基础模型和通用 AI 中涌现吗？ChatGPT 的首次评估](https://arxiv.org/abs/2303.03186)（2023 年 3 月）
- [UZH_CLyp 在 SemEval-2023 任务 9 中：推特亲密度预测的头部优先微调和 ChatGPT 数据生成的跨语言学习](https://arxiv.org/abs/2303.01194)（2023 年 3 月）
- [如何格式化输入到 ChatGPT 模型](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)（2023 年 3 月）
- [ChatGPT 能评估人类个性吗？通用评估框架](https://arxiv.org/abs/2303.01248)（2023 年 3 月）
- [跨语言摘要通过 ChatGPT](https://arxiv.org/abs/2302.14229)（2023 年 2 月）
- [ChatAug：利用 ChatGPT 进行文本数据增强](https://arxiv.org/abs/2302.13007)（2023 年 2 月）
- [Dr ChatGPT，告诉我我想听的：提示知识如何影响健康回答的正确性](https://arxiv.org/abs/2302.13793)（2023 年 2 月）
- [ChatGPT 在数学文字问题（MWP）上的独立评估](https://arxiv.org/abs/2302.13814)（2023 年 2 月）
- [ChatGPT：2.5 个月后的元分析](https://arxiv.org/abs/2302.13795)（2023 年 2 月）
- [让我们聊天！与 ChatGPT 的对话：技术、应用和限制](https://arxiv.org/abs/2302.13817)（2023 年 2 月）
- [检查你的事实并再试一次：通过外部知识和自动反馈改进大型语言模型](https://arxiv.org/abs/2302.12813)（2023 年 2 月）
- [ChatGPT 的鲁棒性：对抗性和超分布视角](https://arxiv.org/abs/2302.12095)（2023 年 2 月）
- [生成 AI 模型如 ChatGPT 如何（误用）在 SPC 实践、教育和研究中？探索性研究](https://arxiv.org/abs/2302.10916)（2023 年 2 月）
- [ChatGPT 也能理解吗？ChatGPT 与微调 BERT 的比较研究](https://arxiv.org/abs/2302.10198)（2023 年 2 月）
- [增强 Prompt 工程与 ChatGPT 的 Prompt 模式目录](https://arxiv.org/abs/2302.11382)（2023 年 2 月）
- [通过与 ChatGPT 聊天进行零样本信息提取](https://arxiv.org/abs/2302.10205)（2023 年 2 月）
- [ChatGPT：样样通，样样松](https://arxiv.org/abs/2302.10724)（2023 年 2 月）
- [ChatGPT 和 DALL-E 2 在决策和空间推理上的试点评估](https://arxiv.org/abs/2302.09068)（2023 年 2 月）
- [关于 AI 的网民、学者和信息专业人士的意见，特别是关于 ChatGPT](https://arxiv.org/abs/2302.07136)（2023 年 2 月）  
- [ChatGPT 中的语言模糊性分析](https://arxiv.org/abs/2302.06426)（2023 年 2 月）  
- [ChatGPT 与传统知识图谱问答：当前状态和朝向知识图谱聊天机器人的未来方向](https://arxiv.org/abs/2302.06466)（2023 年 2 月）  
- [ChatGPT 和生成性 AI 对科学意味着什么](https://www.nature.com/articles/d41586-023-00340-6)（2023 年 2 月）  
- [应用 BERT 和 ChatGPT 进行莱姆病情感分析的科学文献](https://arxiv.org/abs/2302.06474)（2023 年 2 月）  
- [探索 ChatGPT 的 AI 伦理：诊断分析](https://arxiv.org/abs/2301.12867)（2023 年 1 月）  
- [ChatGPT 有益吗？关于大型语言模型在教育中的机会和挑战](https://www.edu.sot.tum.de/fileadmin/w00bed/hctl/_my_direct_uploads/ChatGPT_for_Good_.pdf)（2023 年 1 月）  
- [对话式 AI 的政治意识形态：关于 ChatGPT 的亲环境、左翼自由主义倾向的汇聚证据](https://arxiv.org/abs/2301.01768)（2023 年 1 月）  
- [提高可靠性的技术 - OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)  
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)  
- [介绍 ChatGPT](https://openai.com/blog/chatgpt)（2022 年 11 月）  
   
---  
[上一节（应用）](./prompts-applications.md)  
   
[下一节（对抗性 Prompting）](./prompts-adversarial.md)