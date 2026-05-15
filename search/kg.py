import streamlit as st
from pyvis.network import Network

from services.graph_service import get_case_detail, get_case_graph_data


node_color_map = {
    "案件": "#FF6347",
    "妗堜欢": "#FF6347",
    "人物": "#1E90FF",
    "浜虹墿": "#1E90FF",
    "机构": "#20B2AA",
    "鏈烘瀯": "#20B2AA",
    "地点": "#3CB371",
    "鍦扮偣": "#3CB371",
    "工具": "#FFA500",
    "宸ュ叿": "#FFA500",
    "诈骗类型": "#BA55D3",
    "璇堥獥绫诲瀷": "#BA55D3",
    "实体资产": "#FFD700",
    "瀹炰綋璧勪骇": "#FFD700",
    "罪名": "#A9A9A9",
    "缃悕": "#A9A9A9",
    "法律法规": "#CD853F",
    "娉曞緥娉曡": "#CD853F",
}

rel_color_map = {
    "涉及被害人": "#FF69B4",
    "涉案被害人": "#FF69B4",
    "涉及嫌疑人": "#00CED1",
    "涉案嫌疑人": "#00CED1",
    "属于组织": "#7B68EE",
    "所在地": "#32CD32",
    "案发地点": "#FF4500",
    "触犯法律法规": "#8B0000",
    "诈骗类型": "#9400D3",
    "涉案工具": "#FF8C00",
    "人物关系": "#4682B4",
    "涉案资产": "#9ACD32",
    "罪名": "#808080",
    "刑事判决": "#DC143C",
    "赔偿金额": "#00FA9A",
    "赔偿给": "#00BFFF",
}


def init_net():
    net = Network(
        directed=True,
        height="800px",
        width="100%",
        notebook=False,
        cdn_resources="in_line",
    )
    net.set_options(
        """
        {
            "physics": {
                "enabled": true,
                "stabilization": {"enabled": true, "iterations": 100},
                "timestep": 0.5,
                "adaptiveTimestep": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 0.5
                }
            },
            "interaction": {
                "tooltipDelay": 200,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false
            }
        }
        """
    )
    return net


def visualize_case_network(case_name, net=None):
    graph_data = get_case_graph_data(case_name)
    if not graph_data.success:
        st.warning(graph_data.error.message if graph_data.error else "未找到相关案件信息。")
        return net

    if not net:
        net = init_net()

    existing_nodes = {node["id"] for node in net.nodes}
    for node in graph_data.entities:
        if node.id in existing_nodes:
            continue
        title = "\n".join(f"{key}: {value}" for key, value in node.properties.items())
        net.add_node(
            node.id,
            label=node.label,
            title=title,
            color=node_color_map.get(node.type, "#888888"),
            font={"size": 12},
        )
        existing_nodes.add(node.id)

    existing_edges = {(edge["from"], edge["to"], edge.get("label")) for edge in net.edges}
    for rel in graph_data.relations:
        edge_key = (rel.source, rel.target, rel.type)
        if edge_key in existing_edges or rel.source not in existing_nodes or rel.target not in existing_nodes:
            continue
        net.add_edge(
            rel.source,
            rel.target,
            label=rel.type,
            color=rel_color_map.get(rel.type, "#666666"),
            width=1.5,
            arrows="to",
        )
        existing_edges.add(edge_key)
    return net


def show_net(net, height=500):
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height)


@st.dialog("案件详情", width="large")
def show_case_detail(case_name):
    case = get_case_detail(case_name)
    if not case:
        st.warning("未找到相关案件信息。")
        return

    with st.spinner("加载案件详情中..."):
        st.write(f"案件名称: {case['name']}")
        st.write(f"案件描述: {case['description']}")
        with st.expander("查看判决书", expanded=False):
            st.write(case.get("content", "无判决书信息"))
        with st.spinner("加载知识图谱中..."):
            net = visualize_case_network(case_name)
            if net:
                show_net(net, height=500)
