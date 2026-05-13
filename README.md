# OmniFraud
启动项目终端指令：
终端一：启动 Neo4j
cd E:\haixia\OmniFraud\OmniFraud
.\run_neo4j_console.ps1
终端二：启动 Streamlit 前端
cd E:\haixia\OmniFraud\OmniFraud
.\.venv\Scripts\Activate.ps1
streamlit run app.py

## 介绍

OmniFraud 平台聚焦“防诈于未然，解诈于未解”的设计理念，面向广大普通公众，尤其是老年人、青少年、初入职场者等诈骗易感群体，融合多源数据与前沿智能模型，提供识别—理解—预警—应对四位一体的反诈服务方案，同时也可广泛服务于公安机关、社区治理与教育宣传等反诈应用场景。

### 五大核心功能模块：

1. **知识图谱智能问答**：基于 Neo4j 图数据库与 Qwen-VL 大语言模型构建诈骗案件知识图谱，用户可通过自然语言提问，实现对典型案件的结构化理解与深度问答；
2. **关键词案件检索与图谱可视化**：支持多维标签与关键词驱动的案件检索，结合图谱可视化，帮助用户清晰掌握诈骗流程与关键节点；
3. **反诈资讯推荐系统**：聚合并推送高质量的反诈新闻、案例文章与科普内容，打造个性化信息防线；
4. **诈骗风险自评问卷**：用户可通过问卷评估自身受骗风险，系统自动生成图表分析结果，并结合 Qwen-VL 提供定制化防骗建议；
5. **诈骗短信智能识别系统**：支持用户粘贴短信内容，利用华为 NEZHA 模型识别是否为诈骗短信、所涉诈骗类型，并即时生成应对方案。

## 使用方式

1. 安装所需要的依赖
   ```bash
   pip install -r requirements.txt
   ```

   如有需要，您可以使用 `conda` 或 `venv` 创建一个新的虚拟环境。
2. 运行 Streamlit 服务
   ```bash
   streamlit run app.py
   ```
3. 访问本地服务
   打开浏览器，访问 `http://localhost:8501` 即可使用本平台。若您在云端运行本平台，则将在 8501 端口提供服务，您可以通过云端地址访问。

## 目录结构

1. `Anti-fraud_KG_Construct/`：反诈知识图谱构建模块，包含 Qwen-VL 数据预处理、知识图谱实体提取、RAG 构建
2. `Fraud-msg_Classification/`：反诈短信智能识别模块，包含 NEZHA 模型训练与推理
3. `app.py` Streamlit 网页入口
4. `start_page.py` 欢迎页
5. `recognize/`, `search/`, `show/`, `bot/`：五大模块的依赖文件
6. `recognize_page.py`, `risk_page.py`, `show_page.py`, `bot_page.py`, `search_page.py`：五大模块的页面文件
   ============================

## FastAPI API

在不影响原有 Streamlit 入口的前提下，项目新增了 FastAPI 后端接口，可用于服务化调用与前后端分离集成。

- Streamlit 启动：`streamlit run app.py`
- FastAPI 启动：`python -m uvicorn api.main:app --reload`

当前已提供的核心 API：

- `GET /health`
- `POST /sms/recognize`
- `POST /qa/chat`
- `POST /cases/search`
- `POST /risk-assessment/report`
- `POST /agent/run`

运行测试：

```bash
python -m unittest discover -s tests -v
```
