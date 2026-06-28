# CKK Portfolio Builder v13.2

Adds:
- Slate Rating + Bankroll Coach
- Manual number of games on slate
- Bankroll/risk preference inputs
- Vegas Impact table by matchup
- Optional The Odds API key support for live MLB totals/moneylines
- Vegas Score included in Stack Trend Score
- Stack Confidence + Stack Recommendation

Run:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Notes:
- Weather is API-based through Open-Meteo using DK Salaries Game Info/home team ballpark mapping.
- Vegas defaults to neutral if no Odds API key is entered.
- Current Vegas support uses current totals/moneylines. Opening-line movement can be added once a source/API that exposes openers is connected.
