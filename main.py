import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# ======================== 数据定义 ========================
words_embedding = {
    "狮子": [1.0, 0.0, 0.2, 0.9, 0.95, 0.85, 0.4],
    "老虎": [1.0, 0.0, 0.2, 0.88, 0.96, 0.9, 0.35],
    "猎豹": [1.0, 0.0, 0.1, 0.7, 0.7, 1.0, 0.3],
    "大象": [1.0, 0.0, 0.1, 1.0, 0.6, 0.3, 0.5],
    "长颈鹿": [1.0, 0.0, 0.0, 0.95, 0.2, 0.5, 0.6],
    "猫": [1.0, 0.0, 0.1, 0.3, 0.2, 0.7, 0.7],
    "狗": [1.0, 0.0, 0.1, 0.4, 0.3, 0.65, 0.85],
    "狼": [1.0, 0.0, 0.1, 0.6, 0.8, 0.8, 0.2],
    "马": [1.0, 0.0, 0.1, 0.75, 0.2, 0.9, 0.5],
    "苹果": [0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.8],
    "香蕉": [0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.85],
    "橙子": [0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.82],
    "草莓": [0.0, 1.0, 0.0, 0.15, 0.0, 0.0, 0.9],
    "胡萝卜": [0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.6],
    "西兰花": [0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.5],
    "国王": [0.0, 0.0, 0.95, 0.7, 0.5, 0.2, 0.6],
    "女王": [0.0, 0.0, -0.95, 0.7, 0.5, 0.2, 0.7],
    "王子": [0.0, 0.0, 0.9, 0.5, 0.3, 0.4, 0.65],
    "公主": [0.0, 0.0, -0.9, 0.45, 0.2, 0.4, 0.8],
    "男人": [0.0, 0.0, 0.85, 0.6, 0.2, 0.4, 0.5],
    "女人": [0.0, 0.0, -0.85, 0.55, 0.15, 0.4, 0.7],
    "汽车": [0.0, 0.0, 0.0, 0.6, 0.4, 0.85, 0.3],
    "火车": [0.0, 0.0, 0.0, 0.85, 0.3, 0.6, 0.2],
    "飞机": [0.0, 0.0, 0.0, 0.9, 0.2, 1.0, 0.4],
    "船": [0.0, 0.0, 0.0, 0.7, 0.1, 0.4, 0.3],
    "玫瑰": [0.0, 0.9, 0.0, 0.1, 0.0, 0.0, 0.95],
    "向日葵": [0.0, 0.9, 0.0, 0.4, 0.0, 0.0, 0.88],
    "河流": [0.0, 0.0, 0.0, 0.5, 0.0, 0.3, 0.4],
    "山脉": [0.0, 0.0, 0.0, 0.95, 0.2, 0.0, 0.5],
    "云": [0.0, 0.0, 0.0, 0.3, 0.0, 0.6, 0.4],
    "火焰": [0.0, 0.0, 0.0, 0.3, 0.9, 0.7, 0.2],
    "冰": [0.0, 0.0, 0.0, 0.3, 0.2, 0.1, -0.2],
    "快乐": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "悲伤": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9],
    "愤怒": [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, -0.7],
    "雄狮": [1.0, 0.0, 0.95, 0.92, 0.96, 0.84, 0.45],
    "母狮": [1.0, 0.0, -0.95, 0.85, 0.94, 0.8, 0.5],
    "程序员": [0.0, 0.0, 0.6, 0.2, 0.0, 0.2, 0.4],
    "设计师": [0.0, 0.0, 0.55, 0.2, 0.0, 0.2, 0.7],
    "熊猫": [1.0, 0.0, 0.0, 0.6, 0.2, 0.3, 0.9],
    "企鹅": [1.0, 0.0, 0.0, 0.4, 0.1, 0.5, 0.7],
    "葡萄": [0.0, 1.0, 0.0, 0.15, 0.0, 0.0, 0.86],
    "摩托车": [0.0, 0.0, 0.0, 0.5, 0.3, 0.9, 0.5],
}

words = list(words_embedding.keys())
vectors = np.array([words_embedding[w] for w in words])
dim_names = ["动物性🐾", "植物性🌿", "性别倾向⚥", "体型📏", "危险度⚠️", "速度⚡", "情感❤️"]


# ======================== 工具函数 ========================
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0


def most_similar(word, top_k=5):
    idx = words.index(word)
    target_vec = vectors[idx]
    sims = [(w, cosine_similarity(target_vec, vectors[i])) for i, w in enumerate(words) if w != word]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def analogy(a, b, c):
    idx_a = words.index(a)
    idx_b = words.index(b)
    idx_c = words.index(c)
    result_vec = vectors[idx_a] - vectors[idx_b] + vectors[idx_c]
    best_word = None
    best_sim = -1
    for i, w in enumerate(words):
        if w in (a, b, c):
            continue
        sim = cosine_similarity(result_vec, vectors[i])
        if sim > best_sim:
            best_sim = sim
            best_word = w
    return best_word, best_sim


# ======================== Dash 应用 ========================
app = dash.Dash(__name__)
app.title = "词嵌入可视化实验室"

# 下拉菜单选项
dim_options = [{"label": dim_names[i], "value": i} for i in range(len(dim_names))]

app.layout = html.Div([
    html.H1("🧠 词嵌入空间实验室", style={"textAlign": "center", "color": "#2c3e66"}),
    html.P("点击任意单词查看详细信息与相似词 | 选择坐标轴维度探索语义空间", style={"textAlign": "center"}),

    html.Div([
        # 左侧图形区
        html.Div([
            dcc.Graph(id="scatter-plot", config={"displayModeBar": True}, style={"height": "600px"}),
            html.Div([
                html.Label("X轴:", style={"fontWeight": "bold"}),
                dcc.Dropdown(id="x-dim", options=dim_options, value=0, clearable=False,
                             style={"width": "180px", "display": "inline-block"}),
                html.Label("Y轴:", style={"fontWeight": "bold", "marginLeft": "20px"}),
                dcc.Dropdown(id="y-dim", options=dim_options, value=3, clearable=False,
                             style={"width": "180px", "display": "inline-block"}),
            ], style={"textAlign": "center", "marginTop": "10px"}),
        ], style={"width": "68%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

        # 右侧信息面板
        html.Div([
            html.H3("📌 当前单词", style={"marginBottom": "5px"}),
            html.Div(id="selected-word-info",
                     style={"background": "#f0f2f6", "padding": "12px", "borderRadius": "10px", "minHeight": "150px"}),
            html.H3("🔗 最相似单词", style={"marginTop": "20px", "marginBottom": "5px"}),
            html.Div(id="similar-words",
                     style={"background": "#f0f2f6", "padding": "12px", "borderRadius": "10px", "minHeight": "150px"}),
            html.H3("🧪 向量类比", style={"marginTop": "20px"}),
            html.Div([
                html.Div([
                    html.Label("A:", style={"width": "30px"}),
                    dcc.Dropdown(id="analogy-a", options=[{"label": w, "value": w} for w in words], value="国王",
                                 clearable=False),
                ], style={"display": "inline-block", "width": "30%"}),
                html.Div([
                    html.Label("B:", style={"width": "30px"}),
                    dcc.Dropdown(id="analogy-b", options=[{"label": w, "value": w} for w in words], value="男人",
                                 clearable=False),
                ], style={"display": "inline-block", "width": "30%", "marginLeft": "5%"}),
                html.Div([
                    html.Label("C:", style={"width": "30px"}),
                    dcc.Dropdown(id="analogy-c", options=[{"label": w, "value": w} for w in words], value="女人",
                                 clearable=False),
                ], style={"display": "inline-block", "width": "30%", "marginLeft": "5%"}),
                html.Button("计算类比 (A - B + C)", id="analogy-btn", n_clicks=0,
                            style={"marginTop": "15px", "width": "100%", "backgroundColor": "#3498db", "color": "white",
                                   "border": "none", "padding": "8px", "borderRadius": "5px"}),
                html.Div(id="analogy-result",
                         style={"background": "#f0f2f6", "marginTop": "15px", "padding": "10px", "borderRadius": "10px",
                                "minHeight": "80px"}),
            ]),
        ], style={"width": "28%", "display": "inline-block", "verticalAlign": "top", "padding": "10px",
                  "marginLeft": "10px"}),
    ]),
    html.Div(id="selected-word-store", style={"display": "none"}),
    html.Footer("💡 提示：点击散点图中的单词点即可选中，右侧会显示其向量和相似词。",
                style={"textAlign": "center", "marginTop": "20px", "color": "#7f8c8d", "fontSize": "0.9rem"})
])


# ======================== 回调函数 ========================
@app.callback(
    Output("selected-word-store", "children"),
    Input("scatter-plot", "clickData"),
    prevent_initial_call=True
)
def store_selected_word(clickData):
    """存储点击的单词"""
    if clickData and "points" in clickData:
        word = clickData["points"][0]["customdata"][0]
        return word
    return dash.no_update


@app.callback(
    Output("scatter-plot", "figure"),
    Input("x-dim", "value"),
    Input("y-dim", "value"),
    Input("selected-word-store", "children")
)
def update_scatter(x_idx, y_idx, selected_word):
    """更新散点图，高亮选中的点"""
    x_vals = vectors[:, x_idx]
    y_vals = vectors[:, y_idx]

    # 构建 hover 文本
    hover_texts = []
    for i, w in enumerate(words):
        vec_str = "<br>".join([f"{dim_names[d]}: {vectors[i][d]:.2f}" for d in range(len(dim_names))])
        hover_texts.append(f"<b>{w}</b><br>{vec_str}")

    # 颜色和大小：选中点为红色，其余为蓝色
    colors = ['red' if w == selected_word else 'steelblue' for w in words]
    sizes = [12 if w == selected_word else 8 for w in words]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers+text',
        text=words,
        textposition="top center",
        textfont=dict(size=10, color="black"),
        marker=dict(color=colors, size=sizes, line=dict(width=1, color='white')),
        hovertext=hover_texts,
        hoverinfo='text',
        customdata=[[w] for w in words]  # 用于点击获取单词
    ))

    fig.update_layout(
        title=f"词嵌入空间投影：{dim_names[x_idx]} vs {dim_names[y_idx]}",
        xaxis_title=dim_names[x_idx],
        yaxis_title=dim_names[y_idx],
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    return fig


@app.callback(
    Output("selected-word-info", "children"),
    Input("selected-word-store", "children")
)
def update_selected_info(selected_word):
    """显示选中单词的向量详情（换行显示）"""
    if not selected_word:
        return "点击左侧图中的任意单词开始探索"
    idx = words.index(selected_word)
    vec = vectors[idx]
    # 为每个维度创建一个 Div 元素，自动换行
    vec_items = [html.Div(f"{dim_names[i]}: {vec[i]:.2f}") for i in range(len(dim_names))]
    return html.Div([
        html.H4(selected_word, style={"margin": "0 0 8px 0", "color": "#e67e22"}),
        html.Div(vec_items, style={"fontSize": "0.9rem"})
    ])


@app.callback(
    Output("similar-words", "children"),
    Input("selected-word-store", "children")
)
def update_similar_words(selected_word):
    """显示最相似的单词列表"""
    if not selected_word:
        return "暂无"
    sims = most_similar(selected_word, 5)
    return html.Div([
        html.Div([
            html.Span(f"{w}: {s:.3f}", style={"marginRight": "15px", "padding": "4px 8px", "background": "#dfe6e9",
                                              "borderRadius": "20px", "display": "inline-block", "marginBottom": "5px"})
            for w, s in sims
        ])
    ])


@app.callback(
    Output("analogy-result", "children"),
    Input("analogy-btn", "n_clicks"),
    State("analogy-a", "value"),
    State("analogy-b", "value"),
    State("analogy-c", "value"),
    prevent_initial_call=True
)
def update_analogy(n_clicks, a, b, c):
    """计算并显示类比结果"""
    if n_clicks == 0:
        return "点击按钮进行计算"
    result_word, sim = analogy(a, b, c)
    return html.Div([
        html.Div(f"{a} - {b} + {c}  ≈  {result_word}", style={"fontWeight": "bold", "fontSize": "1.1rem"}),
        html.Div(f"相似度: {sim:.3f}", style={"color": "#2c3e66", "marginTop": "5px"})
    ])


if __name__ == "__main__":
    print("启动 Dash 应用... 请在浏览器中打开 http://127.0.0.1:8050")
    app.run(debug=True)  # 修改此处
