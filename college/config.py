"""
NCAAB Betting Model Configuration
"""

NCAAB_SEASONS = [2022, 2023, 2024, 2025, 2026]  # ESPN season year = ending year
NCAAB_DB_PATH = "data/ncaab_betting.db"
NCAAB_FEATURES_CSV = "data/ncaab_training_features.csv"
NCAAB_MODELS_DIR = "models/saved/ncaab/"
HOME_COURT_ADVANTAGE = 3.5  # pts, larger than NBA
ROLLING_WINDOW = 10  # games
MIN_GAMES_REQUIRED = 5  # minimum games before including team in features

# ESPN API base URLs
ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_TEAMS = f"{ESPN_BASE}/teams"
ESPN_RANKINGS = f"{ESPN_BASE}/rankings"

# Odds API
ODDS_API_KEY = "c8e21d1a5e018412eebbdc966e970ef8"
ODDS_API_NCAAB = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"

# Rate limiting
REQUEST_DELAY = 0.3  # seconds between requests
