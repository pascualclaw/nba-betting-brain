"""
Props backtesting — loads historical prop data and scores what the model would have predicted.

Usage:
    python3 backtesting/props_backtest.py
    python3 backtesting/props_backtest.py --verbose
"""
import sys
import json
from pathlib import Path
from datetime import datetime

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.props_model import PropsPredictor
from config import DATA_DIR

PROPS_DIR = DATA_DIR / "props"
PREDICTOR = PropsPredictor()


def load_prop_history() -> list:
    """Load all prop JSON files from data/props/."""
    if not PROPS_DIR.exists():
        return []
    
    props = []
    for f in sorted(PROPS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            # Only load player props (not game totals/spreads)
            if data.get("player") and data.get("prop_type"):
                props.append(data)
        except Exception as e:
            print(f"  ⚠️  Failed to load {f.name}: {e}")
    
    return props


def backtest_prop(prop: dict, verbose: bool = False) -> dict:
    """
    Run the model against a historical prop and score it.
    
    Returns:
        {
            prop_id, player, prop_type, line, direction,
            actual_value, hit (actual result),
            model_predicted, model_recommendation, model_confidence, model_edge,
            model_correct (did model's rec match actual result),
            model_direction_correct (did model say OVER/UNDER correctly),
        }
    """
    player = prop.get("player", "")
    prop_type = prop.get("prop_type", "")
    line = float(prop.get("line", 0))
    actual_value = prop.get("actual_value")
    hit = prop.get("hit")  # True = actual went over
    direction = prop.get("direction", "over").lower()
    
    # Parse team/opponent from game_code: "PHX_SAC_20260303"
    game_code = prop.get("game_code", "")
    parts = game_code.split("_")
    team = parts[0] if len(parts) >= 2 else ""
    opponent = parts[1] if len(parts) >= 2 else ""
    date = f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}" if len(parts) >= 3 and len(parts[2]) == 8 else ""
    
    if verbose:
        print(f"\n  Backtesting: {player} {prop_type} {direction} {line}")

    try:
        result = PREDICTOR.predict_prop(
            player=player,
            team=team,
            opponent=opponent,
            prop_type=prop_type,
            line=line,
            date=date,
        )
    except Exception as e:
        return {
            "prop_id": prop.get("prop_id"),
            "player": player,
            "error": str(e),
            "model_correct": None,
        }

    if "error" in result:
        return {
            "prop_id": prop.get("prop_id"),
            "player": player,
            "error": result["error"],
            "model_correct": None,
        }

    model_rec = result["recommendation"]
    model_pred = result["predicted"]
    model_conf = result["confidence"]
    model_edge = result["edge"]

    # Actual outcome
    actual_over = hit  # True if the prop hit (went over)
    
    # Did model's direction match?
    model_says_over = model_rec == "OVER"
    model_says_under = model_rec == "UNDER"
    model_says_pass = model_rec == "PASS"
    
    model_direction_correct = None
    if actual_over is not None and not model_says_pass:
        model_direction_correct = (model_says_over == actual_over)
    
    # Model "correct" = it said OVER and it hit, or UNDER and it didn't hit
    model_correct = None
    if actual_over is not None and not model_says_pass:
        model_correct = model_direction_correct

    return {
        "prop_id": prop.get("prop_id"),
        "player": player,
        "prop_type": prop_type,
        "line": line,
        "direction": direction,
        "actual_value": actual_value,
        "actual_over": actual_over,
        "model_predicted": round(model_pred, 2),
        "model_recommendation": model_rec,
        "model_confidence": round(model_conf, 3),
        "model_edge": round(model_edge, 2),
        "model_direction_correct": model_direction_correct,
        "model_correct": model_correct,
        "model_reasoning": result.get("reasoning", ""),
    }


def run_backtest(verbose: bool = False) -> dict:
    """
    Run backtest on all historical props.
    
    Returns:
        {
            total, scored, correct, wrong, passed,
            accuracy (% of non-pass that were correct),
            results: [list of individual backtest results]
        }
    """
    print(f"\n{'='*60}")
    print(f"📊 Props Model Backtest")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    props = load_prop_history()
    if not props:
        print(f"\n  ⚠️  No prop history found in {PROPS_DIR}")
        return {"total": 0, "results": []}
    
    print(f"\n  Loaded {len(props)} historical props")
    
    results = []
    errors = 0
    
    for prop in props:
        r = backtest_prop(prop, verbose=verbose)
        results.append(r)
        if r.get("error"):
            errors += 1
            if verbose:
                print(f"  ❌ Error on {r['player']}: {r['error']}")
    
    # Score
    scored = [r for r in results if r.get("model_correct") is not None]
    correct = [r for r in scored if r.get("model_correct") == True]
    wrong = [r for r in scored if r.get("model_correct") == False]
    passed = [r for r in results if r.get("model_recommendation") == "PASS"]
    
    accuracy = len(correct) / len(scored) if scored else 0
    
    print(f"\n  Results:")
    print(f"    Total props: {len(props)}")
    print(f"    Errors (couldn't predict): {errors}")
    print(f"    Model said PASS: {len(passed)}")
    print(f"    Model made predictions: {len(scored)}")
    print(f"    Correct: {len(correct)}")
    print(f"    Wrong: {len(wrong)}")
    print(f"    Accuracy: {accuracy*100:.1f}%")
    
    print(f"\n  Individual results:")
    for r in results:
        status = "✅" if r.get("model_correct") == True else \
                 "❌" if r.get("model_correct") == False else \
                 "⏭️ " if r.get("model_recommendation") == "PASS" else "❓"
        print(f"    {status} {r.get('player', '?'):20s} {r.get('prop_type', '?'):10s} "
              f"line={r.get('line', 0):5.1f} "
              f"pred={r.get('model_predicted', 0):5.1f} "
              f"actual={r.get('actual_value', '?')!s:5} "
              f"rec={r.get('model_recommendation', '?'):5s} "
              f"conf={r.get('model_confidence', 0)*100:.0f}%")
        if verbose and r.get("model_reasoning"):
            print(f"         {r['model_reasoning'][:100]}...")
    
    print(f"\n{'='*60}\n")
    
    return {
        "total": len(props),
        "scored": len(scored),
        "correct": len(correct),
        "wrong": len(wrong),
        "passed": len(passed),
        "errors": errors,
        "accuracy": round(accuracy, 3),
        "results": results,
    }


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_backtest(verbose=verbose)
