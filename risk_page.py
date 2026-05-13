import streamlit as st
import pandas as pd
import plotly.express as px

from clients.llm_client import LlmClient
from schemas.risk_schema import RiskProfileInput
from services.risk_service import generate_risk_report, generate_risk_report_stream

st.markdown(
    """
<style>
    /* 主标题动画 */
    @keyframes titleAnimation {
        0% { transform: translateY(-20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    /* 主标题 */
    .main-title {
        color: #F1C40F;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #F1C40F;
        animation: titleAnimation 0.5s ease-out;
    }
    
    /* 输入框美化 */
    .stTextInput>div>div>input {
        border-radius: 15px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(255,223,107,0.2);
    }
    
    /* 动态结果卡片 */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    /* 诈骗结果样式 */
    .fraud-result {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
    }
    
    /* 正常结果样式 */
    .normal-result {
        background: linear-gradient(135deg, #63cdda, #77ecb9);
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.expander("📌 操作说明", expanded=True):
        st.markdown(
        """
        1. **⚠️ 风险评估系统**：根据用户填写的问卷信息，生成个性化的风险分析报告和建议。
        2. **📝 风险分析报告**：总结用户的风险特征，并给出实用建议。
        3. **📈 指标关联分析**：展示用户信息与风险值之间的关联热力图和路径分析图。
        """
        )
    with st.expander("⚙️ 高级选项"):
        st.header("配置 Qwen-VL API Key")
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
            import openai
            with st.spinner('正在验证...'):
                try:
                    openai.base_url = st.session_state.openai_base_url
                    openai.api_key = st.session_state.openai_api_key
                    openai.models.retrieve(st.session_state.openai_model)
                    st.success('API Key 验证成功', icon='✅')
                except Exception as e:
                    st.error(e, icon='❌')

def get_openai_response(user_profile):
    profile_str = "\n".join([f"{k}: {v}" for k, v in user_profile.items()])
    prompt_template = """
    我这里有一个用户对风险评估问卷填入的信息，以下是信息内容：
    {profile_str}

    请你根据用户的信息，为用户生成一个风险分析报告，并给予用户实用建议。

    要求：
    1. 总结用户对风险评估问卷的回答，分析用户的风险特征。
    2. 对用户的信息先给出具体详细的分析，再给出建议。
    3. 可以针对用户信息中的内容的部分特征，给出风险用户建议。
    4. 只需要给出约 200 字的建议即可。建议有条理地列出。
    5. 适量加入 emoji 表情，使得建议更加生动有趣。
    6. 分点回答时 emoji 表情或 icon 不要放在句子结尾。
    7. 不要输出任何其它的内容，只输出风险分析报告与建议内容。

    风险分析报告与建议内容：
    """
    prompt = prompt_template.format(profile_str=profile_str)
    client = OpenAI()

    response = client.chat.completions.create(
        model=st.session_state.openai_model,
        messages=[
            {
                "role": "system",
                "content": "The following is a message that I received from a user and I need your help to respond to it.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=1.0,
        stream=True
    )
    return response


def get_openai_response(user_profile):
    return generate_risk_report_stream(user_profile)

def risk_assessment_page():
    # ========== 页面配置 ==========
    # ========== 核心功能布局 ==========
    st.markdown("<h1 class='main-title'>⚠️ 反诈风险评估系统</h1>",unsafe_allow_html=True)

    # ========== 核心功能布局 ==========
    with st.form("main_form"):
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            with st.expander("🔍 个人信息画像", expanded=True):
                age = st.slider(
                    "年龄",
                    18,
                    100,
                    25,
                    help="不同年龄段风险特征：\n- 青年(18-30)：网络诈骗高风险\n- 中年(31-50)：投资理财诈骗敏感\n- 老年(51+)：保健品诈骗易感",
                )
                residence = st.selectbox(
                    "常住地区",
                    ["城市", "县城", "农村"],
                    help="根据2023年反诈白皮书，农村地区金融诈骗报案率高出城市23%",
                )
                occupation = st.selectbox(
                    "职业类型",
                    ["学生", "在职员工", "自由职业", "退休人员", "其他"],
                    help="自由职业者遭遇兼职刷单诈骗的概率是其他职业的2.1倍",
                )
                income = st.selectbox("💴 月收入范围", ["无固定收入", "3000元以下", "3000-8000元", "8000-20000元", "20000元以上"])

            # 新增金融行为分析
            with st.expander("💳 金融行为分析", expanded=True):
                payment_methods = st.multiselect(
                    "常用支付方式（多选）",
                    ["刷脸支付", "微信支付", "银行卡支付", "动态令牌"],
                    default=["微信支付"],
                )
                investment_experience = st.selectbox(
                    "投资经验",
                    ["无", "1年以下", "1-3年", "3年以上"],
                    help="有投资经验的用户更易受到投资诈骗",
                )

        with col2:
            with st.expander("📈 风险接触分析", expanded=True):
                # 诈骗类型增加权重标识
                fraud_types = st.multiselect(
                    "近半年接触的诈骗类型（多选）",
                    [
                        ("冒充公检法"),
                        ("投资理财"),
                        ("网络购物"),
                        ("兼职刷单"),
                        ("感情诈骗"),
                        ("中奖诈骗"),
                        ("健康养生"),
                    ],
                    format_func=lambda x: x,
                    default=[("网络购物")],
                )

                loss_amount = st.slider("近一年被诈骗金额（元）", 0, 200000, 0, 1000)
                report_police = st.checkbox("是否及时报警", value=True)

            with st.expander("💡 心理评估", expanded=True):
                st.markdown("**遇到以下情况时您的反应：**")
                col_a, col_b, col_c = st.columns([1, 1, 1], gap="medium")

                with col_a:
                    urgency_react = st.radio(
                        "🕒 收到'紧急'通知时",
                        ("立即查看", "先核实再处理", "直接忽略"),
                        index=1,
                        help="研究表明80%的诈骗利用紧急心理"
                    )

                with col_b:
                    stranger_request = st.radio(
                        "👤 陌生人请求个人信息",
                        ("婉言拒绝", "视情况而定", "爽快提供"),
                        index=0,
                        help="信息泄露是诈骗的主要源头"
                    )

                with col_c:
                    reward_react = st.radio(
                        "🎁 面对未验证的优惠信息",
                        ("果断举报", "保持怀疑", "积极参与"),
                        index=1,
                        help="高回报承诺通常是诈骗诱饵"
                    )
            social_media = st.slider("每日社交媒体使用时长(小时)", 
                                   0, 24, 3,
                                   help="过度暴露个人信息增加风险")

        submitted = st.form_submit_button(
            "开始风险评估", use_container_width=True, type="primary"
        )

    # ========== 评估结果可视化 ==========
    if submitted:
        user_profile = {
            "年龄": age,
            "地区": residence,
            "职业": occupation,
            "月收入范围": income,
            "支付方式": payment_methods,
            "投资经验": investment_experience,
            "接触诈骗类型": fraud_types,
            "近一年被诈骗金额": loss_amount,
            "是否报警": report_police,
            "紧急反应": urgency_react,
            "陌生人处理": stranger_request,
            "未验证优惠信息": reward_react,
            "每日社交使用时长(小时)": social_media,
        }
        # with st.spinner("🤯 正在评估中..."):
        #     try:
        #         # 生成风险分析报告
        #         response = get_openai_response(user_profile)
        #     except Exception as e:
        #         st.error(f"❌ 评估失败，请稍后再试。错误信息：{e}")
        #         st.stop()
        # print(user_profile["接触诈骗类型"])
        # st.success("✅ 评估完成！正在为您生成风险分析报告🫡")
        tag = st.empty()
        tab1, tab2 = st.tabs(["📝 风险分析报告", "📊 指标关联分析"])
        with tab1:
            with st.spinner("🤯 正在评估中..."):
                try:
                    # 生成风险分析报告
                    response = get_openai_response(user_profile)
                    st.write_stream(response)
                except Exception as e:
                    st.error(f"❌ 评估失败，请稍后再试。错误信息：{e}")
                    st.stop()
            # st.write(user_profile)
            tag.success("✅ 评估完成！已为您生成风险分析报告🫡")
            st.toast(":rainbow[结果已就绪！]", icon="🎉")
            st.balloons()
        with tab2:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    "<h3 style='text-align: center;'>风险因子关联分析热力图</h3>",
                    unsafe_allow_html=True,
                )

                # 数据预处理
                sample_data = pd.DataFrame({
                    "年龄": [28, 35, 22, 45, 31, 27, 50, 38, 29, 33],
                    "地区编码": [0, 0, 1, 2, 0, 1, 2, 0, 0, 1],
                    "收入等级": [3, 4, 2, 2, 3, 3, 1, 4, 2, 3],
                    "社交时长": [5, 3, 7, 2, 4, 6, 1, 3, 5, 4],
                    "诈骗类型数": [3, 1, 2, 0, 2, 4, 1, 2, 3, 1],
                    "心理评估分": [65, 82, 48, 73, 70, 55, 60, 85, 68, 58],
                    "风险值": [85, 68, 92, 58, 75, 88, 63, 70, 82, 78]
                })

                # 计算相关系数矩阵
                corr_matrix = sample_data.corr(method='spearman')

                # 生成热力图
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale='RdBu_r',
                    labels=dict(color="相关系数"),
                    width=650,
                    height=650
                )
                fig.update_layout(
                    xaxis=dict(tickangle=45, tickfont=dict(size=10)),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                fig.update_traces(
                    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>相关系数: %{z:.2f}",
                    hoverongaps=False
                )

                # 可视化呈现
                st.plotly_chart(fig, use_container_width=True)

                # 数据表格展示
                with st.expander("📜 原始相关系数矩阵"):
                    styled_matrix = corr_matrix.style\
                        .background_gradient(cmap='RdBu_r', vmin=-1, vmax=1)\
                        .format("{:.2f}")\
                        .set_table_styles([{
                            'selector': 'th',
                            'props': [('font-size', '10pt'), 
                                    ('background-color', '#f8f9fa')]
                        }])
                    st.dataframe(styled_matrix, use_container_width=True)

                with c2:
                    st.markdown(
                        "<h3 style='text-align: center;'>多变量关联路径分析图</h3>",
                        unsafe_allow_html=True,
                    )

                    # 扩展数据
                    parallel_df = pd.DataFrame({
                        '年龄': [28, 35, 22, 45, 31, 27, 50, 38, 29, 33],
                        '收入等级': [3, 4, 2, 2, 3, 3, 1, 4, 2, 3],
                        '社交时长': [5, 3, 7, 2, 4, 6, 1, 3, 5, 4],
                        '诈骗类型数': [3, 1, 2, 0, 2, 4, 1, 2, 3, 1],
                        '心理评估分': [65, 82, 48, 73, 70, 55, 60, 85, 68, 58],
                        '风险值': [85, 68, 92, 58, 75, 88, 63, 70, 82, 78]
                    })

                    # 创建平行坐标图
                    fig = px.parallel_coordinates(
                        parallel_df,
                        color="风险值",
                        color_continuous_scale=px.colors.diverging.Tealrose,
                        labels={
                            "年龄": "Age",
                            "收入等级": "Income Level",
                            "社交时长": "Social Time",
                            "诈骗类型数": "Fraud Types",
                            "心理评估分": "Psychological Score",
                            "风险值": "Risk Value"
                        },
                        height=750,
                    )
                    # 调整图表边距，使布局更加紧凑
                    fig.update_layout(
                        margin=dict(l=80, r=50, t=80, b=50),
                        font=dict(size=13),
                        xaxis=dict(
                            tickangle=45,
                            tickfont=dict(size=10, color="black"),  # 设置刻度颜色为黑色
                        ),
                        yaxis=dict(
                            tickfont=dict(size=10, color="black")  # 设置刻度颜色为黑色
                        ),
                    )

                    # 显示平行坐标图
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("请填写所有信息后点击开始风险评估查看结果😴")

risk_assessment_page()
