import streamlit as st

from clients.neo4j_client import Neo4jClient
from search import kg
from services.case_service import get_case_names as get_case_names_service
from services.case_service import get_schema_summary
from services.case_service import search_cases as search_cases_service


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
        border-radius: 20px;
        padding: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">知识图谱检索</h1>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def search_cases(keyword: str, skip: int = 0, limit: int = 30):
    response = search_cases_service(keyword, limit=limit, skip=skip)
    if not response.success:
        raise ValueError(response.error.message if response.error else "案件检索失败")
    return response.total_count, response.cases


def get_cases_names(limit: int = 5):
    return get_case_names_service(limit=limit)


def render_schema_diagnostic():
    try:
        schema_rows = get_schema_summary(limit=12)
    except Exception as exc:
        st.error(f"Neo4j schema 诊断失败：{exc}")
        return

    if not schema_rows:
        st.warning("Neo4j 已连接，但数据库中暂未发现任何节点。")
        return

    st.caption("当前 Neo4j 节点标签统计：")
    st.dataframe(schema_rows, use_container_width=True, hide_index=True)


def render_case_card(case, index: int):
    title = case.title or "未命名案件"
    case_type = case.case_type or "未知类型"
    with st.expander(f"{case_type}案件：{title}", expanded=True):
        st.markdown(f"#### {case.summary or '暂无案件描述'}")
        if case.fraud_types or case.fraud_subtypes:
            fraud_text = " - ".join(
                part for part in (", ".join(case.fraud_types), ", ".join(case.fraud_subtypes)) if part
            )
            st.markdown(f"**类型**: {fraud_text}")
        if case.suspects:
            st.markdown(f"**嫌疑人**: {', '.join(case.suspects)}")
        if case.victims:
            st.markdown(f"**被害人**: {', '.join(case.victims)}")
        if case.money:
            st.markdown(f"**涉案金额**: {case.money:,.2f} 元")
        if case.locations:
            st.markdown(f"**地点**: {', '.join(case.locations)}")
        if case.laws:
            st.markdown(f"**法律法规**: {', '.join(case.laws)}")
        if st.button("查看详情", key=f"view_kg_{index}", use_container_width=True):
            kg.show_case_detail(title)


with st.sidebar:
    with st.expander("操作说明", expanded=True):
        st.markdown(
            """
        1. 输入关键词检索案件。
        2. 点击推荐案件或搜索结果查看详情。
        3. 下方会自动展示推荐案件的知识图谱。
        """
        )

    with st.expander("高级选项"):
        st.header("Neo4j 数据库连接配置")
        use_custom_neo4j = st.checkbox("自定义 Neo4j 连接配置")

        if use_custom_neo4j:
            st.session_state.neo4j_uri = st.text_input("Neo4j URL")
            st.session_state.neo4j_username = st.text_input("Neo4j 用户名")
            st.session_state.neo4j_database = st.text_input("Neo4j 数据库")
            st.session_state.neo4j_password = st.text_input("Neo4j 密码", type="password")
        else:
            st.session_state.neo4j_uri = st.secrets["NEO4J_URI"]
            st.session_state.neo4j_username = st.secrets["NEO4J_USERNAME"]
            st.session_state.neo4j_database = st.secrets["NEO4J_DATABASE"]
            st.session_state.neo4j_password = st.secrets["NEO4J_PASSWORD"]

        if st.button("检查连接可用性"):
            with st.spinner("正在连接..."):
                try:
                    Neo4jClient().verify_connectivity()
                    st.success("连接成功")
                except Exception as exc:
                    st.error(exc)


keyword = st.text_input("请输入关键词进行搜索：", "")
search_clicked = st.button("开始搜索", key="search_btn", use_container_width=True, type="primary")

if search_clicked or keyword.strip():
    if not keyword.strip():
        st.warning("请输入有效的关键词进行搜索。")
    else:
        with st.spinner("正在搜索..."):
            try:
                total_count, cases = search_cases(keyword)
                if total_count <= 0 or not cases:
                    st.info("没有找到匹配的案件。")
                    render_schema_diagnostic()
                else:
                    st.success(f"共找到 {total_count} 条匹配案件，当前展示 {len(cases)} 条。")
                    for index, case in enumerate(cases):
                        render_case_card(case, index)
            except Exception as exc:
                st.error(f"搜索时发生错误：{exc}")
                render_schema_diagnostic()
else:
    try:
        with st.expander("智能推荐案件", expanded=True):
            with st.spinner("载入推荐案件..."):
                cases_names = get_cases_names(limit=4)
                if not cases_names:
                    st.info("暂无可推荐案件，请检查 Neo4j 中是否已有案件数据。")
                    render_schema_diagnostic()
                else:
                    cols = st.columns(2)
                    for index, case_name in enumerate(cases_names):
                        with cols[index % 2]:
                            st.button(
                                case_name,
                                use_container_width=True,
                                key=f"case_{index}",
                                help="点击查看案件详情",
                                on_click=kg.show_case_detail,
                                args=(case_name,),
                            )

        with st.expander("知识图谱可视化案件", expanded=True):
            with st.spinner("载入知识图谱..."):
                if not cases_names:
                    st.info("暂无案件可用于绘制知识图谱。")
                else:
                    net = kg.init_net()
                    for case_name in cases_names:
                        try:
                            net = kg.visualize_case_network(case_name, net)
                        except Exception as exc:
                            st.toast(f"加载案件 {case_name} 时发生错误：{exc}")

                    if net.nodes:
                        kg.show_net(net, height=800)
                        st.toast("知识图谱加载完成。")
                    else:
                        st.info("已找到案件，但暂未查询到可展示的图谱节点。")
    except Exception as exc:
        st.error(f"推荐案件加载失败：{exc}")
        render_schema_diagnostic()
