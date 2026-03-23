# 词嵌入交互式可视化应用 (Dash 版本)

基于 Dash 和 Plotly 构建的词嵌入空间可视化工具，支持维度选择、单词点击、相似词查询和向量类比实验。

## 功能特性
- **动态散点图**：可任意选择 X/Y 轴维度（共 7 个语义维度），实时更新词向量投影。
- **单词详情**：点击散点图中的单词点，右侧显示其完整向量值和最相似的 5 个单词。
- **向量类比**：输入三个单词 A、B、C，计算 `A - B + C` 的结果，并返回最匹配的单词及相似度。

## 安装与运行

### 环境要求
- Python 3.7+
- pip

### 1. 克隆仓库
```bash
git clone https://github.com/Wang-yeqi/word-embedding-lab.git
cd word-embedding-lab
```
### 2. 配置环境
```bash
pip install -r requirements.txt
```
## 项目结构
.
-├── app.py                # 主程序
-├── requirements.txt      # 依赖列表
-└── README.md             # 项目说明
## 自定义词向量
当前词向量为示例数据（共 40 个单词，7 个维度）。如需替换，请修改 main.py 中的 words_embedding 字典和 dim_names 列表。
