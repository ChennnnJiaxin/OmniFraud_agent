import streamlit as st

from infra.config import ensure_streamlit_session_config, load_app_config

st.set_page_config(
    page_title="OmniFraud",
    layout="wide",
    page_icon="assets/logo.png",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)

ensure_streamlit_session_config(load_app_config())

st.logo("assets/OmniFraud2.png", size="large", icon_image="assets/OmniFraud2.png")

start_page = st.Page("start_page.py", title="欢迎", icon="🏠")
recognize_page = st.Page("recognize_page.py", title="短信识别", icon="📡")
agent_page = st.Page("agent_page.py", title="统一反诈智能体", icon="🧭")
bot_page = st.Page("bot_page.py", title="问答助手", icon="🤖")
risk_page = st.Page("risk_page.py", title="风险评估", icon="📳")
search_page = st.Page("search_page.py", title="案件搜索", icon="🔍")
show_page = st.Page("show_page.py", title="反诈警示", icon="📰")

pages = [start_page, recognize_page, agent_page, bot_page, risk_page, search_page, show_page]
pg = st.navigation(pages)
pg.run()
