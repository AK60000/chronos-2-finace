# Chronos-2 股票预测系统

[**在线体验 / Live Demo**](http://cwj0.ysjohnson.top/stock/)

基于亚马逊 Chronos-2 时间序列大模型的股票价格预测工具。集成了新浪财经实时数据抓取功能，并提供交互式的 Streamlit 仪表盘。

## 系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│    新浪财经     │────>│    数据抓取器    │────>│     预处理器     │────>│  Chronos-2 模型  │
│    HTTP API     │     │  (日线 OCHLV)    │     │  (NumPy 数组)    │     │  (1.2亿参数)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                    │
                                                                                    v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Plotly 图表   │<────│    结果解析器    │<────│    预测 API     │
│  (K线/成交量)   │     │      (分位数)    │     │  (Tensor 输出)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 功能特性

- **零样本预测 (Zero-shot)**：无需微调。预训练的 Chronos-2 模型可直接对未见过的股票数据生成预测。
- **多变量协变量支持**：除了收盘价目标序列外，还使用开盘价、最高价、最低价和成交量作为历史协变量。
- **概率性预测**：输出 21 个分位数水平（0.01 到 0.99），通过置信区间实现不确定性量化。
- **交互式可视化**：基于 Plotly 的蜡烛图，包含移动平均线（MA5/MA20）、成交量柱状图和预测覆盖层。
- **自动市场检测**：支持沪市（6xxxxx）和深市股票代码，自动解析前缀。
- **设备无关推理**：自动检测并利用可用 GPU，或降级到 CPU 运行。

## 模型规格

| 参数 | 数值 |
|-----------|-------|
| 架构 | 基于 T5 的编码器 (Chronos2Model) |
| 参数量 | 120M (1.2亿) |
| 上下文长度 | 8,192 个时间步 |
| 最大预测长度 | 1,024 个时间步 |
| 输入补丁大小 | 16 |
| 输出补丁大小 | 16 |
| 分位数水平 | 21个 (0.01, 0.05, 0.1, ..., 0.95, 0.99) |
| 归一化方法 | arcsinh |

## 输入 / 输出

### 输入

| 维度 | 说明 |
|-----------|-------------|
| 目标 (Target) | 收盘价 (日线) |
| 历史协变量 | 开盘价、最高价、最低价、成交量 (日线) |

### 输出

| 维度 | 说明 |
|-----------|-------------|
| 预测值 | 未来 N 个交易日的收盘价预测 |
| 分位数 | 每个时间步的 21 个概率水平（可配置显示的子集） |

## 项目结构

```
chronos-2/
├── chronos-2/              # 预训练模型权重 (本地)
│   ├── config.json
│   └── model.safetensors
├── app.py                  # Streamlit Web 界面
├── scraper.py              # 新浪财经数据抓取器
├── preprocessor.py         # 数据预处理工具
├── predictor.py            # Chronos-2 推理封装
├── requirements.txt        # Python 依赖项
└── .gitignore
```

## 快速开始

### 环境要求

- Python 3.10+
- 支持 PyTorch 的 Conda 环境

### 安装

```bash
conda activate GPU
pip install -r requirements.txt
```

### 运行

```bash
streamlit run app.py
```

应用程序将运行在 `http://localhost:8501`。

## 使用说明

1. 在侧边栏输入股票代码（例如：贵州茅台 `600519`）
2. 调整历史数据范围和预测周期
3. 点击 "开始预测" (Start Prediction) 获取数据并进行推理
4. 在交互式仪表盘查看结果：
   - 带有 MA5/MA20 的历史蜡烛图
   - 成交量图表
   - 带有 80% 置信区间的预测覆盖层

## 部署

### 服务器部署 (CPU)

当 GPU 不可用时，模型会自动回退到 CPU。生产环境部署：

```bash
# 安装依赖
pip install -r requirements.txt

# 使用 Streamlit 运行
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

可以添加 Dockerfile 进行容器化部署。模型权重（`model.safetensors`，约 240MB）应包含在镜像中或通过卷挂载。

## 数据来源

历史日线 K 线数据获取自新浪财经 HTTP API：

- 接口地址：`https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData`
- 覆盖范围：A 股市场（沪市 + 深市）
- 频率：日线
- 字段：日期、开盘价、最高价、最低价、收盘价、成交量

## 免责声明

本工具仅供教育和研究目的使用，不构成投资建议。股票市场投资具有风险，过往业绩不代表未来表现。模型的预测是概率估计，不应作为投资决策的唯一依据。

## 参考资料

- [Amazon Chronos-2 Model Card](https://huggingface.co/amazon/chronos-2)
- [Chronos Forecasting GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Technical Paper](https://arxiv.org/abs/2510.15821)

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 作者

[AK60000](https://github.com/AK60000)
