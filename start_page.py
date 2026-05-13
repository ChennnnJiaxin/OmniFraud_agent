import streamlit as st

with st.sidebar:
    with st.expander("🧰 **各个页面功能说明**", expanded=True):
        st.markdown(
        """
        1. **🎉 欢迎页面**：展示应用特色及视频演绎。
        2. **📩 短信识别页面**：对短信进行诈骗识别，提取关键信息并给出建议。
        3. **🤖 问答助手**：基于知识图谱的问答助手，提供智能反诈问答服务。
        4. **📊 风险评估页面**：对用户填写的问卷进行风险评估，给出相应的风险分析报告和个性化建议。
        5. **🔍 案件搜索页面**：根据构建的知识图谱进行多维度案件搜索，提供案件详情和相关信息。
        6. **📰 反诈警示页面**：展示热门和最新的反诈警示与知识，提供文章详情和阅读数据。
        """
        )

st.title(":rainbow[OmniFraud]")
st.markdown("### :rainbow[*融合 Qwen-VL 多模态识别与知识图谱研判的反诈赋能 Agent*]")
video_path = "assets/logo.mp4"
st.video(video_path, autoplay=True, muted=True, loop=True)
