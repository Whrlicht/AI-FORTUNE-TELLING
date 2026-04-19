# AI Fortune Telling (FastAPI + LangChain + RAG)

这是一个可扩展的初始框架，包含：

- FastAPI 接口服务
- LangChain 在线 LLM 对话
- Chroma 本地向量库（RAG）
- 空知识库初始化（可先不导入文档）

## 1. 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填写你的在线 LLM API Key：

```bash
copy .env.example .env
```

至少需要设置：

- `OPENAI_API_KEY`

如果你使用的是 OpenAI 兼容平台（例如代理网关或第三方模型平台），可修改：

- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

## 3. 启动服务

```bash
uvicorn app.main:app --reload
```

启动后可访问：

- `GET /health`
- `GET /api/v1/rag/status`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`

## 4. 调用示例

### 查看当前 RAG 文档数量

```bash
curl http://127.0.0.1:8000/api/v1/rag/status
```

### 对话（启用 RAG）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"我今天适合做什么？\",\"use_rag\":true}"
```

当知识库为空时，系统会退化为纯 LLM 对话；后续你可以在 `app/services/rag_service.py` 中扩展导入文档逻辑。

### 流式对话（Windows CMD + curl）

先切换到 UTF-8，避免中文输出乱码：

```bat
chcp 65001
```

然后用 `curl` 调用流式接口：

```bat
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream -H "Content-Type: application/json" -d "{\"message\":\"你好，请分3点自我介绍\",\"use_rag\":true}"
```

说明：

- `-N` 表示关闭客户端缓冲，更容易看到逐段输出。
- 知识库为空时，`use_rag=true` 仍可正常流式回答，只是不会引用知识库来源。
