# 本地文档问答（local-doc-qa-agent）

一个不依赖向量库、不依赖云服务的本地文档问答脚本：解析 `.pptx` / `.docx` / `.txt` / `.md`，
切块后用 BM25 检索，再把命中的原文交给本地大模型（Ollama）或任意 OpenAI 兼容接口生成回答。
没配模型时退化为「检索摘要」模式，照样能用。

当前目录里的两份文档就是内置样例：

- `答辩展示.pptx` —— 在线教育离线数仓 0-1 构建，72 页
- `林鑫垚 - 副本.docx` —— 简历

## 快速开始

```bash
pip install -r requirements.txt      # 只装 jieba（可选，装了中文分词更准）
python qa.py                          # 进入交互式问答
```

环境里没有 Python 时，Windows 可直接安装：

```powershell
winget install --id Python.Python.3.12 --exact
```

## 常用命令

```bash
# 单次提问
python qa.py "项目的周期是多长"

# 只查某几份文档（可重复，也可用逗号分隔）
python qa.py --docs "林鑫垚 - 副本.docx" "她的邮箱是什么"
python qa.py --docs "答辩展示.pptx,林鑫垚 - 副本.docx" "数据仓库分了哪几层"

# 只看检索结果，不调用大模型
python qa.py --no-llm "增量同步是怎么做的"

# 查看索引切成了哪些块
python qa.py --list
```

交互模式下支持 `/sources`（看索引统计）和 `/exit`（退出）。

## 接入大模型

按优先级自动选择，都不存在时退回检索摘要。

**1. Ollama（推荐，完全本地）**

```bash
ollama pull qwen2.5:7b
ollama serve
```

```bash
python qa.py "项目的目标是什么"
python qa.py --model qwen2.5:7b "项目指导老师是谁"
```

可选环境变量：`OLLAMA_HOST`（默认 `http://127.0.0.1:11434`）、`OLLAMA_MODEL`。

**2. OpenAI 兼容接口（DeepSeek 等）**

```powershell
$env:DEEPSEEK_API_KEY="sk-..."         # 或 OPENAI_API_KEY
$env:OPENAI_MODEL="deepseek-chat"      # 可选
```

## 工作原理

1. **抽取**：直接读 OOXML（`zipfile` + `ElementTree`），不依赖 `python-pptx` / `python-docx`；
   PPT 一页一块，Word 的碎段落按长度合并。
2. **切块**：超过 420 字按句子边界切分，相邻块保留 80 字重叠。
3. **检索**：jieba 搜索模式分词 + BM25；整串命中加权、短块轻微降权、相邻页扩散、近似重复页去重。
4. **生成**：把 top-k 原文拼成 Prompt，要求模型只依据资料回答并标注 `[编号]` 来源。

索引结果缓存在 `.qa_index.json`，源文件改动或切块逻辑升级后自动重建。

## 已知限制

- 检索是关键词级别的，纯语义问题（如资料里只写了「Zabbix」，却问「用什么监控工具」）可能召回不到，
  接入大模型后也只能基于召回到的片段回答。
- 只支持 `pptx/docx/txt/md`，PDF 需要先转成文本。
- 扫描版文档（纯图片）没有文字层，无法解析。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `qa.py` | 全部逻辑：解析、切块、检索、生成、CLI |
| `requirements.txt` | 唯一可选依赖 jieba |
| `.qa_index.json` | 自动生成的索引缓存（已加入 .gitignore） |
