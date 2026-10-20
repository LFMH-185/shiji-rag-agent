# 《史记》智能问答（shiji-rag-agent）

基于《史记》全文的检索增强问答系统：本地 CPU 完成解析、切块、向量化与混合检索，
调用 DeepSeek 流式生成**白话解释 + 原文引用 + 卷篇出处**的回答；检索置信度不足时直接拒答，不编造。

- 语料：130 篇 / 625,478 字 / 2,082 个文本块
- 检索：Dense（bge-small-zh）+ BM25 + RRF，纯 CPU，单次 50–70 ms
- 评测：**Recall@5 = 0.950**，Recall@3 = 0.750，MRR = 0.578（30 题，含 10 条负样本）
- 应用：CLI（`qa.py`）与 Web（`app.py`，Streamlit，多轮对话 + 流式输出）

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [数据与切块](#数据与切块)
- [检索设计](#检索设计)
- [问答层](#问答层)
- [评测结果](#评测结果)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [性能与成本](#性能与成本)
- [常见问题](#常见问题)
- [已知限制](#已知限制)

---

## 功能特性

| 能力 | 说明 |
|---|---|
| 章节级定位 | 每个文本块都带「卷号 / 篇名 / 篇型 / 段落区间 / 字符偏移」，回答可回溯原文 |
| 混合检索 | 向量召回 + BM25 字面召回，RRF 融合，兼顾语义与专名 |
| 文言适配 | 异体字归一（阬/坑、於/于、適/适…）、典故扩展（焚书坑儒 → 焚诗书/阬术士） |
| 上下文补全 | 命中块前后各取 1 块，避免断章取义 |
| 强约束生成 | 只依据检索资料作答；无记载时回复「《史记》未见记载」；每条结论带 `[n]` 引用 |
| 置信度门控 | `dense < 0.65 且 BM25 无命中` 时本地拒答，不消耗 API |
| 流式输出 | DeepSeek SSE 流式返回，CLI 与 Web 均逐字渲染 |
| 多轮对话 | Web 端保留上下文；短追问先独立检索，置信不足才合并上一轮问题 |
| 零配置界面 | 无任何输入框：接口/模型/参数全部内置，Key 只读环境变量 |
| 防刷 | 提问后输入框冷却 2 秒，避免连点刷请求 |
| 可复现评测 | `evaluate.py` + `eval/questions.jsonl`，一条命令跑出 Recall / MRR |

---

## 系统架构

```text
                    ┌──────────────────────────────┐
   shiji.txt ──────▶│ preprocess.py                │
   （UTF-8 全文）    │ 卷篇解析 → 段落聚合 → 递归切块 │
                    └──────────────┬───────────────┘
                                   │ data/chunks.jsonl（2,082 块，含元数据）
                                   ▼
                    ┌──────────────────────────────┐
                    │ index.py                     │
                    │ bge-small-zh-v1.5 → 512 维    │
                    │ Chroma（cosine）              │
                    └──────────────┬───────────────┘
                                   │ data/chroma
                                   ▼
   用户问题 ──▶ retrieve.py ────────────────────────────┐
                   ├─ Dense：Chroma top-30              │
                   ├─ BM25：jieba 搜索分词 top-30       │
                   └─ RRF（k=10，权重 Dense:BM25=1:0.2）│
                                   │                    │
                                   ▼                    │
                         top-5 + 前后各 1 邻块            │
                                   │                    │
                                   ▼                    │
                    ┌──────────────────────────────┐    │
                    │ qa.py / app.py               │    │
                    │ 门控 → Prompt → DeepSeek 流式 │◀───┘
                    └──────────────┬───────────────┘
                                   ▼
                 白话解释 + 原文引用 [n] + 卷篇出处
```

### 模块职责

| 文件 | 职责 |
|---|---|
| `preprocess.py` | 解析卷/篇结构、清洗、递归切块、生成 `chunks.jsonl` |
| `fix_data.py` | 把缺失篇目按格式补入语料，并做异体字对齐 |
| `index.py` | 向量化入库（Chroma），带增量判断与自测 |
| `retrieve.py` | 混合检索、查询扩展、邻块扩展、可选 rerank |
| `evaluate.py` | 评测 Recall@k / MRR、负样本拒答率 |
| `qa.py` | 检索 + Prompt + DeepSeek 流式问答（CLI） |
| `app.py` | Streamlit Web 界面（多轮对话、流式、引用展开） |

---

## 数据与切块

### 语料

- 底本：`shiji.txt`（UTF-8，625,478 字，6,247 行）
- 结构：目录 130 条 + 正文 130 篇（本纪 12 / 表 10 / 书 8 / 世家 30 / 列传 70）
- 已修复：底本缺失的卷四十三（赵世家）、卷四十四（魏世家）、卷四十八（陈涉世家），由 `fix_data.py` 补入
- 用字对齐：补丁中的「于/适/谪/强/後/鄴/沉/溪/剪/暗/伙/扣」统一为主库写法「於/適/適/彊/后/邺/沈/谿/翦/闇/夥/叩」

### 切块

```python
RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["。", "！", "？", "；", "\n\n", "\n", "，", ""],
    keep_separator="end",
)
```

- 先按「卷 → 段 → 句」分层，`太史公曰` 强制独立成块
- 句末标点优先于换行：段落常超过 80 字，若先按 `\n` 切会丢弃重叠
- 实测：1,999 → **2,082 块**，块长中位数 387，最大 400；相邻块重叠率约 90%

### 元数据

```json
{
  "id": "shiji-v006-c019",
  "volume_no": 6,
  "volume": "卷六",
  "chapter": "秦始皇本纪第六",
  "chapter_type": "本纪",
  "chunk_index": 19,
  "para_start": 45,
  "para_end": 46,
  "char_start": 41005,
  "char_end": 41394,
  "is_taishigong": false,
  "text": "丞相李斯曰：……",
  "embed_text": "《史记》卷六·秦始皇本纪第六：丞相李斯曰：……"
}
```

- `text` 用于生成与展示，`embed_text` 额外拼接卷篇前缀后用于向量化
- `char_start/char_end` 可精确回切原文（2,082 块全部通过校验）

---

## 检索设计

### 三路信号

| 通道 | 配置 | 作用 |
|---|---|---|
| Dense | `BAAI/bge-small-zh-v1.5`，512 维，cosine，top-30 | 语义匹配，跨用词差异 |
| BM25 | jieba `lcut_for_search`，k1=1.5，b=0.75，top-30 | 专名、生僻字、原文措辞 |
| 融合 | RRF，k=10，Dense:BM25 = **1 : 0.2** | Dense 主导，BM25 补位 |

> 实测结论：Dense 单独 Recall@5=0.80，BM25 单独 0.40，等权 RRF 反而降到 0.65。
> BM25 的长块噪声会挤掉 Dense 的正确结果，因此权重压到 0.2。

### 文言文适配

1. **异体字归一**（查询与语料两侧同时归一，不改原文）
   `阬→坑`、`馀→余`、`於→于`、`適→适`、`彊→强`、`後→后`、`谿→溪`、`沈→沉`、`翦→剪`、`闇→暗`、`夥→伙`、`叩→扣`
2. **典故扩展**（28 组，仅用于 BM25）
   `焚书坑儒 → 焚诗书 / 阬术士 / 四百六十`、`鸿门宴 → 鸿门 / 项庄 / 樊哙`、`完璧归赵 → 和氏璧 / 蔺相如` …
3. **邻块扩展**：命中块前后各取 1 块，标记为「上下文补充」，不参与排序

### 追问处理

短追问（≤15 字）采用**两段式检索**：

```
先用当前问题独立检索
  ├─ 置信度足够 → 直接使用
  └─ 不足 → 拼接上一轮问题再检索，取更优者
```

> 直接拼接并不总更好：「那扶苏后来怎样了」单独检索命中《蒙恬列传》（BM25 8.4），
> 拼上前文「秦始皇焚书坑儒」后，jieba 把「扶苏」切成「扶/苏」，反而命中《苏秦列传》。

---

## 问答层

### Prompt 约束（要点）

1. 先用**现代白话文**回答，讲清人物、时间、因果，不照抄文言
2. 关键情节引用 1–2 句核心原文，用「」括起并标 `[n]`
3. **禁止只有文言没有解释**
4. 文末列出处：`[1] 《史记·卷六·秦始皇本纪》`
5. 资料没有的，只输出「《史记》未见记载」，禁止用常识/传说/影视剧情补充
6. 全文以白话为主，300 字以内

### 门控

```python
if dense_similarity < 0.65 and bm25_score <= 0:
    return "《史记》未见记载"   # 不调用 API
```

阈值 0.65 由评测集确定：正样本 20/20 通过，空实体负样本 7/7 拒答（0.60 只能拒 6/7）。

---

## 评测结果

```bash
python evaluate.py
```

```
检索评测    : 30 题（正 20 / 负 10）
语料        : 2082 块   RRF top-5
Recall@5    : 0.950  (19/20)
Recall@3    : 0.750  (15/20)
MRR         : 0.578
空实体负样本 : 拒答 7/7
错配负样本   : 单块同时包含实体的 0/3（必须由 LLM 拒答）
```

| 指标 | 数值 | 说明 |
|---|---|---|
| Recall@5 | **0.950** | 19/20 题的前 5 条包含金标准原文 |
| Recall@3 | 0.750 | 15/20 |
| MRR | 0.578 | 未启用 rerank |
| 负样本拒答 | 7/7 | 诸葛亮、杨贵妃、李白、赤壁、郑和、武则天、玄奘 |
| 张冠李戴 | 0/3 误支持 | 司马迁见秦始皇、项羽著兵法、商鞅救刘邦 |

评测集构造：`eval/questions.jsonl`，正样本用「卷号 + 原文片段」定位 gold，
负样本分 `negative_absent`（实体不存在）与 `negative_mismatch`（实体存在但事实错配）。

> 唯一未命中：`q20 司马迁为什么要写《史记》`——BM25 排第 1，但 Dense 排第 40，
> 受权重限制未能进入 top-5，属于语义泛化与字面精确的固有冲突。

---

## 快速开始

### 1. 环境

- Python 3.10+
- Windows / macOS / Linux 均可，纯 CPU（无需 GPU）
- 首次运行需下载约 96 MB 的 embedding 模型

```bash
pip install -r requirements.txt
```

依赖：`torch`(CPU)、`sentence-transformers`、`chromadb`、`modelscope`、`jieba`、`langchain-text-splitters`、`streamlit`

### 2. 语料

把全文放到项目根目录 `shiji.txt`（UTF-8）。该文件体积较大，已在 `.gitignore` 中忽略。

### 3. 预处理与建库

```bash
python preprocess.py          # → data/chunks.jsonl（2,082 块）
python index.py               # → data/chroma（约 30 MB）
```

模型默认从 **ModelScope** 下载（HuggingFace 在部分网络环境不可达）。
也可手动指定本地目录：

```bash
python index.py --model-path D:/models/bge-small-zh-v1.5
```

### 4. 配置 DeepSeek

界面不提供任何 API Key 输入框，只从环境变量读取：

```bash
# Windows（新开终端生效）
setx DEEPSEEK_API_KEY "sk-..."

# macOS / Linux
export DEEPSEEK_API_KEY="sk-..."
```

接口地址固定为 `https://api.deepseek.com`，模型固定为 `deepseek-chat`。

> 未读取到 Key 时，网页只显示「系统维护中，请稍后再试。」，不会引导用户配置任何东西。

### 5. 命令行问答

```bash
python qa.py "秦始皇焚书坑儒"
python qa.py                         # 交互模式
python qa.py "李斯的下场" --show-context
python qa.py "鸿门宴" --dry-run       # 只检索 + 预览 Prompt，不调 API
```

### 6. Web 界面

```bash
python -m streamlit run app.py
```

打开 http://localhost:8501 直接提问即可。

- 左侧边栏只有《史记》简介与使用说明，没有任何配置项
- 支持多轮对话、流式输出、检索依据展开
- 技术参数（阈值/分数）默认折叠在「检索详情（开发者）」中

---

## 目录结构

```text
shiji-rag-agent/
├── preprocess.py           # 解析 + 切块
├── fix_data.py             # 缺失篇目补入与用字对齐
├── index.py                # 向量化入库
├── retrieve.py             # Dense + BM25 + RRF + 邻块
├── evaluate.py             # 评测脚本
├── qa.py                   # CLI 问答（DeepSeek 流式）
├── app.py                  # Streamlit Web 界面
├── requirements.txt
├── eval/
│   ├── questions.jsonl     # 30 题评测集
│   └── report.json         # 评测明细（gitignore）
├── .streamlit/
│   └── config.toml         # 关闭文件监视，避免 torchvision 报错
├── data/                   # 全部 gitignore，可重建
│   ├── chunks.jsonl
│   ├── index_meta.json
│   └── chroma/
├── shiji.txt               # 语料（gitignore）
└── README.md
```

---

## 性能与成本

| 环节 | 耗时（纯 CPU） |
|---|---|
| 预处理（2,082 块） | 约 3 秒 |
| 首次建库（2,082 × 512 维） | 约 2 分 20 秒（14.9 块/秒） |
| 检索（Dense + BM25 + RRF） | 50–70 ms |
| 应用冷启动（加载模型 + BM25） | 约 8 秒 |
| 单次问答上下文 | 约 6 千字（top-5 + 10 个邻块） |

DeepSeek 按输入 token 计费，单次问答成本不到一分钱；拒答路径零 API 调用。

---

## 部署到 Streamlit Cloud

### 1. 仓库准备

代码推送到 GitHub 后，在 [share.streamlit.io](https://share.streamlit.io) 新建应用：

- Repository：`你的账号/shiji-rag-agent`
- Branch：`main`
- Main file path：`app.py`

### 2. 配置 Secrets

进入应用的 **Settings → Secrets**，填入：

```toml
DEEPSEEK_API_KEY = "sk-你的key"
```

保存后点击 **Reboot app**。Streamlit Cloud 会把 Secrets 以环境变量形式注入，
`app.py` 通过 `os.environ.get("DEEPSEEK_API_KEY")` 读取，不需要在界面上填任何东西。

### 3. 语料与索引（自动构建）

`shiji.txt` 已随仓库提交；`data/` 被忽略，由应用在云端**首次启动时自动构建**：

```text
检测 data/chroma → 不存在
  → 页面提示「首次启动，正在构建知识库，请稍候」
  → preprocess.py 切块（约 3 秒）
  → index.py 下载模型并建向量库（约 2–4 分钟）
  → 构建完成后自动进入聊天界面
```

- 构建结果在实例存活期间缓存；重新部署或冷启动会重新构建
- 想跳过构建，可把产物一起提交（约 37 MB）：

```bash
git add -f data/chunks.jsonl data/chroma
git commit -m "Add prebuilt index for faster cold start"
git push
```

- 若构建失败（例如语料缺失），页面只显示「系统维护中」，详细原因在应用日志里

### 4. 资源与冷启动

- 免费实例 1 GB 内存，本方案（bge-small + Chroma）常驻约 400 MB，足够
- 首次加载模型与索引约 8 秒，之后走缓存
- 受限于 Streamlit 免费版算力，首次建库时可能会触发平台保护，等待片刻即可正常使用
- 修改 Secrets 或代码后，记得 **Reboot app**

---

## 常见问题

**Q：Web 页面提示「系统维护中」？**
说明运行环境没有读到 `DEEPSEEK_API_KEY`。本地用 `setx` 配置后新开终端重启；
云端检查 **Settings → Secrets** 是否填写并 Reboot。

**Q：本地提问报 502 / 连接失败？**
接口地址已内置为 `https://api.deepseek.com`，不再由用户填写。
若你之前在代码里改成了本地 mock（`127.0.0.1:8765`）做测试，记得改回并重启。

**Q：终端刷大量 `No module named 'torchvision'`？**
与项目功能无关，是 Streamlit 文件监视器在扫描 `transformers` 的图像处理模块。
本项目已通过 `.streamlit/config.toml` 设置 `fileWatcherType = "none"` 关闭监视。
如仍出现，可用命令行显式关闭：

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

**Q：模型下载失败？**
优先走 ModelScope；也可手动下载后指定：

```bash
python -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-small-zh-v1.5'))"
python index.py --model-path <上一步输出的目录>
```

**Q：修改了 `chunks.jsonl` 后需要重建索引吗？**
需要。`index.py` 通过 `manifest.json` 比对文件指纹，内容变化会自动整库重建；也可 `--rebuild` 强制。

**Q：为什么有的问题回答「《史记》未见记载」？**
可能是检索置信度低于阈值（本地拦截），或资料确实没有记载。CLI 会打印原因，Web 端可在「检索详情（开发者）」查看。

---

## 已知限制

- **检索是关键词 + 语义的混合方案**，对高度抽象的提问（如「司马迁的历史观」）召回有限
- **未启用 rerank**：`bge-reranker-base` 约 1.1 GB，纯 CPU 环境默认关闭（`--enable-rerank` 可开）
- **MRR 0.578**：15 条正样本落在第 2–5 名，排序精度仍有提升空间
- **文言用字**：不同底本存在异体字差异，本项目已做常见归一，但未覆盖全部
- **表格类篇目**（如三代世表）底本为「（表略）」，仅保留标题块
- 语料仅用于学习研究，版权归原作者/出版方所有

---

## License

代码以 MIT 协议开源；`shiji.txt` 等语料不在仓库内，请自行获取并遵守相应版权要求。
