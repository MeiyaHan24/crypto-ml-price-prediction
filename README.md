# Cryptocurrency Price Direction Prediction Using Machine Learning

## Research Question
Can machine learning models (Random Forest, Logistic Regression) predict the
next-day price direction of Bitcoin and Ethereum using technical indicators?

## Dataset
- **Source:** Yahoo Finance via `yfinance` library
- **Assets:** BTC-USD, ETH-USD
- **Period:** January 2022 – January 2025 (daily OHLCV data)
- **Accessed:** April 2026
- **Features:** 10 engineered technical indicators (RSI, MACD, Bollinger Bands, MA Ratio, Volatility, etc.)

## Methods
- Feature engineering (all indicators computed manually using pandas)
- Exploratory data analysis and correlation analysis
- Binary classification: predict next-day price direction (Up=1, Down=0)
- Models: Logistic Regression, Random Forest Classifier
- Time-series train/test split (80/20, chronological order)
- Evaluation: Accuracy, AUC-ROC, confusion matrix, classification report

## Key Findings
*(See notebook Section 6 for full results)*
- Both models perform near the 50% random baseline, consistent with the Efficient Market Hypothesis
- Logistic Regression slightly outperforms Random Forest on both assets
- Short-term momentum (`Price_Change_Pct`) and volatility (`Volatility_7`) are the most predictive features
- Results highlight the difficulty of predicting cryptocurrency price direction using technical indicators alone

## Repository Structure
```
crypto-ml-price-prediction/
├── data/
│   └── crypto_data.csv
├── notebooks/
│   └── crypto_ml_analysis.ipynb
├── outputs/
│   ├── fig1_price_history.png
│   ├── fig2_technical_indicators.png
│   ├── fig3_correlation_heatmap.png
│   ├── fig4_feature_importance.png
│   └── fig5_model_comparison.png
├── README.md
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/crypto_ml_analysis.ipynb
```

## Author
Meiya Han | ACC102 Mini Assignment | XJTLU 2024–25
