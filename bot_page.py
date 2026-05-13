import streamlit as st

from bot.utils import get_session_id, write_message
from clients.llm_client import LlmClient
from clients.neo4j_client import Neo4jClient
from services.qa_service import chat_with_anti_fraud_bot, chat_with_anti_fraud_bot_stream

EXAMPLE_QUESTIONS = [
    "使用手机诈骗的案例有哪些？",
    "有哪些人涉及到了虚假投资？",
    "使用各类工具的诈骗案例分别占比多少？",
    "涉嫌团伙作案的案件有哪些？",
]

st.markdown(
    """
<style>
    @keyframes titleAnimation {
        0% { transform: translateY(-20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }

    .main-title {
        color: #2E86C1;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #2E86C1;
        animation: titleAnimation 0.5s ease-out;
    }

    .stTextInput>div>div>input {
        border-radius: 15px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(255,107,107,0.2);
    }

    .result-card {
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .fraud-result {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
    }

    .normal-result {
        background: linear-gradient(135deg, #63cdda, #77ecb9);
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">🤻 反诈知识问答助手</h1>', unsafe_allow_html=True)

with st.sidebar:
    with st.expander("🎐 操作说明", expanded=True):
        st.markdown(
            """
        1. **🔰 输入问题**：在下方输入框中输入你的问题，点击发送按钮。
        2. **🔳 查看回答**：系统会基于知识图谱和大模型生成回答，并展示在聊天记录中。
        3. **✅ 查看示例问题**：点击上方的示例问题按钮，快速获取常见问题的回答。
        """
        )

    if st.button('重置会话', icon='🔄'):
        st.session_state.messages = [
            {"role": "assistant", "content": "你好，我是关于反诈知识的问答助手。有什么可以帮助到你？🤗"},
        ]
        st.success('会话已重置', icon='✅')

    with st.expander("⚙️ 高级选项"):
        st.header("Qwen-VL API Key 配置")
        use_custom_openai = st.checkbox('自定义 Qwen-VL 连接配置')

        if use_custom_openai:
            st.session_state.openai_api_key = st.text_input('OpenAI API Key', type='password')
            st.session_state.openai_model = st.text_input('OpenAI Model')
            st.session_state.openai_base_url = st.text_input('OpenAI Base URL')
        else:
            st.session_state.openai_api_key = st.secrets['OPENAI_API_KEY']
            st.session_state.openai_model = st.secrets['OPENAI_MODEL']
            st.session_state.openai_base_url = st.secrets['OPENAI_BASE_URL']

        if st.button('检查 API Key 可用性'):
            with st.spinner('正在验证...'):
                try:
                    LlmClient().verify_chat_model()
                    st.success('API Key 验证成功', icon='✅')
                except Exception as exc:
                    st.error(exc, icon='❌')

        st.header("Neo4j 数据库连接配置")
        use_custom_neo4j = st.checkbox('自定义 Neo4j 连接配置')

        if use_custom_neo4j:
            st.session_state.neo4j_uri = st.text_input('Neo4j URL')
            st.session_state.neo4j_username = st.text_input('Neo4j 用户名')
            st.session_state.neo4j_database = st.text_input('Neo4j 数据库')
            st.session_state.neo4j_password = st.text_input('Neo4j 密码', type='password')
        else:
            st.session_state.neo4j_uri = st.secrets['NEO4J_URI']
            st.session_state.neo4j_username = st.secrets['NEO4J_USERNAME']
            st.session_state.neo4j_database = st.secrets['NEO4J_DATABASE']
            st.session_state.neo4j_password = st.secrets['NEO4J_PASSWORD']

        if st.button('检查连接可用性'):
            with st.spinner('正在连接...'):
                try:
                    Neo4jClient().verify_connectivity()
                    st.success('连接成功', icon='✅')
                except Exception as exc:
                    st.error(exc, icon='❌')

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是关于反诈知识的问答助手。有什么可以帮助到你？🤗"},
    ]


def handle_submit(message: str):
    context = {"session_id": get_session_id()}
    try:
        response_stream = chat_with_anti_fraud_bot_stream(message, context=context)
        write_message('assistant', response_stream)
    except Exception:
        response = chat_with_anti_fraud_bot(message, context=context)
        if response.success:
            write_message('assistant', response.answer)
        else:
            error_message = response.error.message if response.error else "抱歉，问答服务暂时不可用。"
            write_message('assistant', error_message)


st.write("试试这些常见问题：")
cols = st.columns(2)
asked_example = None
for index, question in enumerate(EXAMPLE_QUESTIONS):
    with cols[index % 2]:
        if st.button(question, key=f"example_{index}", use_container_width=True):
            asked_example = question

for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

if question := st.chat_input("键入新问题…") or asked_example:
    question = question if question else asked_example
    asked_example = None
    write_message('user', question)
    handle_submit(question)
