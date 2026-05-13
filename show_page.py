from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from schemas.article_schema import ArticleCreateRequest
from services.article_service import get_article_detail, list_articles, publish_article


def apply_page_style():
    st.markdown(
        """
        <style>
        @keyframes titleAnimation {
            0% { transform: translateY(-20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }

        .main-title {
            color: #FFFFFF;
            font-size: 2.5em;
            text-align: center;
            padding: 20px;
            border-bottom: 3px solid #FFFFFF;
            animation: titleAnimation 0.5s ease-out;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: #000000 !important;
        }

        .section-title {
            color: #F1C40F;
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0.3rem 0 0.45rem 0;
        }

        .article-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #F7D76B;
            line-height: 1.3;
            margin-bottom: 0.2rem;
        }

        .article-meta {
            color: #BFC3C9;
            font-size: 0.86rem;
            margin-top: 0;
        }

        .article-preview {
            color: #ECEFF4;
            font-size: 0.95rem;
            line-height: 1.45;
            margin-top: 0.35rem;
            min-height: 4.4em;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .detail-box {
            background: #121212;
            color: #F5F5F5;
            border: 1px solid #2f3640;
            border-radius: 12px;
            padding: 14px 16px;
            line-height: 1.8;
        }

        [data-testid="stMain"] .stButton > button {
            border-radius: 10px;
            border: 1px solid #e0b30a;
            background: linear-gradient(180deg, #f8d449 0%, #f1c40f 100%);
            color: #3b2d00;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown("<h1 class='main-title'>🛡️ 反诈警示中心</h1>", unsafe_allow_html=True)


def publish_article_form():
    with st.form("publish_article_form", clear_on_submit=True):
        title = st.text_input("文章标题", max_chars=50, help="最多50个字符")
        content = st.text_area("文章内容", height=200)
        author = st.text_input("作者", value="匿名")
        is_top = st.checkbox("置顶文章")
        submitted = st.form_submit_button("立即发布")

        if submitted:
            result = publish_article(
                ArticleCreateRequest(
                    title=title,
                    content=content,
                    author=author,
                    is_top=is_top,
                )
            )
            if result.success:
                st.success("✅ 文章发布成功！")
            else:
                st.error(result.error.message if result.error else "文章发布失败")

        if st.form_submit_button("🔧 刷新数据"):
            st.rerun()


def display_article(article, idx: int, is_hot: bool = False):
    icon = "🔥" if is_hot else "📰"
    with st.container(border=True):
        st.markdown(f"<div class='article-title'>{icon} {article.title}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='article-meta'>作者：{article.author} / 发布时间：{article.publish_time}</div>",
            unsafe_allow_html=True,
        )
        preview = (article.content[:88] + "...") if len(article.content) > 88 else article.content
        st.markdown(f"<div class='article-preview'>{preview}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([0.66, 0.34])
        with c1:
            st.progress(article.view_count / 100 if article.view_count < 100 else 1.0, text=f"热度值：{article.view_count}")
        with c2:
            if st.button("阅读全文", key=f"read_{idx}_{article.id}", use_container_width=True):
                st.session_state.selected_article = article.id
                st.rerun()


def show_article_detail(article_id: str):
    article = get_article_detail(article_id)
    if not article:
        st.error("文章不存在")
        st.session_state.pop("selected_article", None)
        return

    st.button("← 返回列表", key="back_btn", on_click=lambda: st.session_state.pop("selected_article"))
    st.title(article.title)
    st.markdown(f"**作者**：{article.author}  |  **发布时间**：{article.publish_time}")
    st.markdown("---")
    st.markdown(f"<div class='detail-box'>{article.content}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align: right; color: #666;'>总阅读量：{article.view_count}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("📱 阅读趋势")
    if article.view_timestamps:
        df = pd.DataFrame({"timestamp": pd.to_datetime(article.view_timestamps)})
        df["date"] = df["timestamp"].dt.floor("D")
        daily_views = df.groupby("date").size().reset_index(name="阅读量")
        fig = px.line(
            daily_views,
            x="date",
            y="阅读量",
            markers=True,
            line_shape="spline",
            template="plotly_white",
            color_discrete_sequence=["#00CC96"],
            labels={"date": "日期", "阅读量": "当日阅读量"},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showline=True, linecolor="lightgray", title="时间", type="date", tickformat="%y-%m-%d"),
            yaxis=dict(showline=True, linecolor="lightgray", title="阅读次数", rangemode="nonnegative"),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=80),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无阅读数据")


def main():
    apply_page_style()

    with st.sidebar:
        st.header("📰 文章管理")
        publish_article_form()

    if "selected_article" in st.session_state:
        show_article_detail(st.session_state.selected_article)
        return

    render_header()
    hot_response = list_articles(sort_by="hot", limit=3)
    latest_response = list_articles(sort_by="latest", limit=10)

    hot_articles = hot_response.articles if hot_response.success else []
    latest_articles = latest_response.articles if latest_response.success else []

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='section-title'>🔥 热门警示</div>", unsafe_allow_html=True)
        if not hot_articles:
            st.info("还没有文章，快去发布一篇吧！")
        for idx, article in enumerate(hot_articles):
            display_article(article, idx, is_hot=True)

    with c2:
        st.markdown("<div class='section-title'>📰 最新资讯</div>", unsafe_allow_html=True)
        if not latest_articles:
            st.info("还没有文章，快去发布一篇吧！")
        for idx, article in enumerate(latest_articles):
            display_article(article, idx + 100, is_hot=False)


main()
