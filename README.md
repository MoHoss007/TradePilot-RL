# TradePilot-RL
# 📈 SARSA-Based Stock Trading Agent

A reinforcement learning project that simulates a stock trading agent using the SARSA algorithm. The agent learns to buy, sell, or hold based on hourly stock prices and technical indicators.

---

## 🚀 Features

- **Trading Environment**: Custom `gym.Env` using hourly stock data from Yahoo Finance.
- **SARSA Agent**: Epsilon-greedy policy with Q-table updates.
- **Technical Indicators**: Includes MACD, RSI, SMA, EMA, and Pivot Points.
- **Training & Testing**: CLI-based training and evaluation with profit tracking.
- **Visualization**: Buy/sell signals plotted on stock price charts for analysis.

---

## 🧠 How It Works

The agent observes a sliding window of stock price differences (with optional technical indicators) and learns an optimal trading strategy using the SARSA algorithm:
  
```
Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]
```

---

## 🛠️ Usage

### ✅ Train the Agent

```bash
python train_sarsa.py   --start_date 2023-06-01   --end_date 2024-06-30   --ticker INTC   --n_episodes 200   --save_dir ./models
```

### 🧪 Test the Agent

```bash
python test_sarsa.py   --start_date 2024-07-01   --end_date 2024-12-30   --ticker INTC   --qtable_path ./models/sarsa_INTC_2023-06-01_to_2024-06-30.pkl
```

---

## 📊 Example Output

- Terminal prints each action (buy/sell/hold), transaction prices, and profits.
- Matplotlib chart displays stock price with buy/sell markers.

---

## 📎 Requirements

- Python 3.8+
- `gym`
- `numpy`
- `matplotlib`
- `pandas`
- `yfinance`

Install dependencies using:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 📈 Experimental Results

Detailed results, charts, and comparisons can be found in the [EXPERIMENTS.md](EXPERIMENTS.md) file.

---

## 🔭 Future Work

- ✅ Implement Deep Q-Network (DQN) to handle high-dimensional continuous state spaces.
- 🔁 Incorporate LSTM to better model temporal patterns in price data.
---

## 📃 License

This project is licensed under the [MIT License](LICENSE).
