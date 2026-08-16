# AI 同人文平台

基于大模型的同人文创作平台，核心目标是**确保人物绝不背离原著**——人物的内外在、言行举止、人际关系必须与原著保持高度一致。

以《魔道祖师》为首个示例，后续可扩展至其他作品。

## 核心特性

### 1. 深度人物档案系统

每个人物包含完整的多维度档案：

| 维度 | 说明 |
|------|------|
| 人物底色 | 最本质的性格内核与核心价值观 |
| 人生经历 | 重要事件与转折点 |
| 人物弧光 | 不同人生阶段的心理状态与深刻变化 |
| 人际关系 | 对不同人的态度、感情定位、相处模式、说话方式 |
| 言行举止 | 性格特征、喜好、口头禅、行为习惯 |
| 绝对禁忌 | 创作中不可违反的行为边界 |
| 原著铁律 | 不可违背的原著事实 |

目前已录入《魔道祖师》5 位核心人物：魏无羡、蓝忘机、江澄、蓝曦臣、温宁。

### 2. 双重 LLM 保障机制

```
用户设定 → 人物档案注入 → LLM 创作 → 一致性审查 → (不合格则) 自动修订 → 输出
```

- **创作阶段**：将完整人物档案 + 关系互动指南 + 时间节点注入 Prompt
- **审查阶段**：独立 LLM 以原著专家身份评分（0-100），检测 OOC、违反原著设定等问题
- **修订阶段**：评分低于 60 时自动修订并重审

### 3. 时间线感知

同一人物在不同阶段状态截然不同——云深不知处的魏无羡 vs 乱葬岗的夷陵老祖 vs 重生后的魏无羡。系统会根据用户选择的时间节点，自动匹配对应的人物弧光阶段。

### 4. 原著文本 RAG 检索

```
剧情设定 + 人物 + 时间节点 → 向量检索 → 注入原著片段 → LLM 创作
```

- **智能分块**：按章节切分原著 txt，500 字一块、100 字重叠，在句号/换行处自然断句
- **人物标注**：入库时自动标注每段涉及的角色名，检索时加权排序
- **多路检索**：基于剧情、人物名、时间节点构建多条 query，合并去重
- **创作注入**：检索到的原著片段作为参考上下文注入 Prompt，引导 LLM 参考原著言行和场景氛围
- **前端展示**：创作结果中展示引用了哪些原著片段及相似度

## 快速开始

### 环境要求

- Python 3.11+
- 任意 OpenAI 兼容 API（OpenAI / DeepSeek / 通义千问 等）

### 安装与运行

```bash
# 1. 进入后端目录
cd backend

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
copy .env.example .env
# 编辑 .env，填入你的 LLM API Key 和模型配置

# 5. 入库原著文本（RAG）
# 将《魔道祖师》全文 txt 放入 backend/app/data/modaozuoshi/text/
venv\Scripts\python scripts\ingest_novel.py

# 6. 启动后端（首次启动会自动入库 demo 文本）
uvicorn app.main:app --reload --port 8000

# 7. 打开前端
# 直接用浏览器打开 frontend/index.html
# 或使用任意静态文件服务器
```

### API 配置示例

```env
# 使用 OpenAI
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# 使用 DeepSeek
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 使用通义千问
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-3-small
```

### RAG / Embedding 配置

| 方案 | 配置 | 说明 |
|------|------|------|
| **A（推荐）** | `EMBEDDING_PROVIDER=local_bge` | 免费本地，与 DeepSeek 搭配零额外费用 |
| **B** | `EMBEDDING_PROVIDER=doubao` | 豆包/火山方舟 API，按量计费 |
| **C** | `EMBEDDING_PROVIDER=deepseek` | DeepSeek 暂无 Embedding，自动回退 local_bge |

**DeepSeek 创作 + 本地 Embedding（推荐）：**

```env
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
EMBEDDING_PROVIDER=local_bge
```

**豆包 Embedding：**

```env
EMBEDDING_PROVIDER=doubao
EMBEDDING_API_KEY=火山方舟APIKey
EMBEDDING_MODEL=ep-xxxxxxxx
EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

接入点 ID 在 [火山方舟控制台](https://console.volcengine.com/ark) 创建 Embedding 模型后获得。

### 原著文本入库

1. 将 UTF-8 编码的 `.txt` 文件放入 `backend/app/data/modaozuoshi/text/`
2. 建议按 `第一章 标题` 格式标注章节
3. 配置 Embedding 并一键入库：

```powershell
cd backend
copy .env.example .env
# 编辑 .env，填入 EMBEDDING_API_KEY（推荐通义 text-embedding-v3）

# 放入 modaozuoshi.txt 后执行
venv\Scripts\python scripts\setup_rag.py
```

测试用 demo 摘要位于 `text/samples/demo_passages.txt`，可通过 `--use-demo` 临时测试：

```powershell
venv\Scripts\python scripts\setup_rag.py --use-demo
```

切换 Embedding 模型或更换文本后，必须 `--force` 重建索引。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（含 RAG 状态） |
| GET | `/api/characters` | 获取所有人物列表 |
| GET | `/api/characters/{id}` | 获取人物完整档案 |
| POST | `/api/stories/create` | 创作同人文（自动 RAG 检索） |
| GET | `/api/rag/status` | RAG 索引状态 |
| POST | `/api/rag/ingest?force=false` | 重新入库原著文本 |
| POST | `/api/rag/search` | 手动检索原著片段 |

### 创作请求示例

```json
{
  "title": "云深不知处的某个夜晚",
  "characters": ["wei_wuxian", "lan_wangji"],
  "scenario": "魏无羡偷偷溜进蓝忘机的房间，想看他睡觉的样子，结果被当场抓住。",
  "time_period": "云深不知处求学时期",
  "tone": "humorous",
  "perspective": "魏无羡第一人称",
  "length": "medium"
}
```

## 项目结构

```
aitongrenwen/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 入口
│   │   ├── config.py               # 配置管理
│   │   ├── models/
│   │   │   ├── character.py        # 人物数据模型
│   │   │   └── story.py            # 故事数据模型
│   │   ├── services/
│   │   │   ├── character_service.py # 人物档案加载与管理
│   │   │   ├── story_generator.py  # 故事生成 + 一致性校验
│   │   │   ├── rag_service.py      # RAG 检索服务
│   │   │   ├── text_chunker.py     # 原著文本分块
│   │   │   └── llm_client.py       # LLM 调用封装
│   │   └── data/
│   │       └── modaozuoshi/
│   │           ├── characters/     # 人物 YAML 档案
│   │           ├── text/           # 原著 txt 文件（用户放入）
│   │           └── chroma/         # 向量索引（自动生成）
│   ├── scripts/
│   │   └── ingest_novel.py         # 原著入库脚本
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html                  # Web 创作界面
└── README.md
```

## 扩展计划

- [ ] 更多《魔道祖师》人物（温情、金光瑶、聂怀桑、金凌等）
- [x] 原著文本 RAG 检索，创作时引用相关原文片段
- [ ] 用户自定义人物档案编辑器
- [ ] 连载故事管理（章节、续写）
- [ ] 社区分享与评论
- [ ] 支持更多作品（天官赐福、杀破狼等）
- [ ] 人物对话模拟（与角色聊天）
- [ ] 多轮互动式故事创作

## 技术架构

```
┌─────────────┐     ┌──────────────────────────────────┐     ┌─────────────┐
│  Web 前端    │────▶│  FastAPI 后端                     │────▶│  LLM API    │
│  (HTML/JS)  │     │                                  │     │  (OpenAI等)  │
└─────────────┘     │  ┌────────────┐  ┌────────────┐ │     └─────────────┘
                    │  │ 人物档案库  │  │ RAG 向量库  │ │     ┌─────────────┐
                    │  │ (YAML)     │  │ (ChromaDB) │ │────▶│ Embedding   │
                    │  └────────────┘  └────────────┘ │     │ API         │
                    │  ┌────────────┐                  │     └─────────────┘
                    │  │ 一致性引擎  │                  │
                    │  └────────────┘                  │
                    └──────────────────────────────────┘
```
