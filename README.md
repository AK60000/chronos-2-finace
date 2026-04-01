# Chronos-2 Stock Prediction System

[**Live Demo / 在线体验**](http://cwj0.ysjohnson.top/stock/)

A stock price forecasting tool built on Amazon's Chronos-2 time series foundation model, featuring real-time data ingestion from Sina Finance and an interactive Streamlit dashboard.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Sina Finance   │────>│   Data Scraper   │────>│  Preprocessor   │────>│  Chronos-2 Model │
│  HTTP API       │     │  (OCHLV Daily)   │     │  (NumPy Arrays) │     │  (120M params)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                   │
                                                                                   v
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Plotly Charts │<────│  Result Parser   │<────│  Predict API    │
│  (K-line/Vol)   │     │  (Quantiles)     │     │  (Tensor Out)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Features

- **Zero-shot forecasting**: No fine-tuning required. The pre-trained Chronos-2 model can generate predictions directly on unseen stock data.
- **Multivariate covariate support**: Uses Open, High, Low, Volume as past covariates alongside the Close target series.
- **Probabilistic predictions**: Outputs 21 quantile levels (0.01 to 0.99), enabling uncertainty quantification via confidence intervals.
- **Interactive visualization**: Plotly-based candlestick charts with moving averages, volume bars, and forecast overlays.
- **Automatic market detection**: Supports both Shanghai (6xxxxx) and Shenzhen stock codes with automatic prefix resolution.
- **Device-agnostic inference**: Automatically detects and utilizes available GPU or falls back to CPU.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | T5-based encoder (Chronos2Model) |
| Parameters | 120M |
| Context length | 8,192 time steps |
| Max prediction length | 1,024 time steps |
| Input patch size | 16 |
| Output patch size | 16 |
| Quantile levels | 21 (0.01, 0.05, 0.1, ..., 0.95, 0.99) |
| Normalization | arcsinh |

## Input / Output

### Input

| Dimension | Description |
|-----------|-------------|
| Target | Close price (daily) |
| Past covariates | Open, High, Low, Volume (daily) |

### Output

| Dimension | Description |
|-----------|-------------|
| Forecast | Close price predictions for N future trading days |
| Quantiles | 21 probability levels per time step (configurable subset for display) |

## Project Structure

```
chronos-2/
├── chronos-2/              # Pre-trained model weights (local)
│   ├── config.json
│   └── model.safetensors
├── app.py                  # Streamlit web interface
├── scraper.py              # Sina Finance data scraper
├── preprocessor.py         # Data preprocessing utilities
├── predictor.py            # Chronos-2 inference wrapper
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Quick Start

### Prerequisites

- Python 3.10+
- Conda environment with PyTorch support

### Installation

```bash
conda activate GPU
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

1. Enter a stock code in the sidebar (e.g., `600519` for Kweichow Moutai)
2. Adjust the historical data range and prediction horizon
3. Click "Start Prediction" to fetch data and run inference
4. View results on the interactive dashboard:
   - Historical candlestick chart with MA5/MA20
   - Volume chart
   - Forecast overlay with 80% confidence interval

## Deployment

### Server Deployment (CPU)

The model automatically falls back to CPU when GPU is unavailable. For production deployment:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

A Dockerfile can be added for containerized deployment. The model weights (`model.safetensors`, ~240MB) should be included in the image or mounted as a volume.

## Data Source

Historical daily K-line data is fetched from the Sina Finance HTTP API:

- Endpoint: `https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData`
- Coverage: A-share market (Shanghai + Shenzhen)
- Frequency: Daily
- Fields: Date, Open, High, Low, Close, Volume

## Disclaimer

This tool is provided for educational and research purposes only. It does not constitute financial advice. Stock market investments carry inherent risks, and past performance does not guarantee future results. The model's predictions are probabilistic estimates and should not be used as the sole basis for investment decisions.

## References

- [Amazon Chronos-2 Model Card](https://huggingface.co/amazon/chronos-2)
- [Chronos Forecasting GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Technical Paper](https://arxiv.org/abs/2510.15821)

## License

This project is licensed under the [MIT License](LICENSE).

## Author

[AK60000](https://github.com/AK60000)
