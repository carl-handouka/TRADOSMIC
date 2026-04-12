
import pandas as pd
import numpy as np
import itertools
import collections


import collections
import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

QUEUE_SHARE_TOUCH = 0
TICK_SIZES = {"EMERALDS": 1, "TOMATOES": 1}
POSITION_LIMITS = {"EMERALDS": 20.0, "TOMATOES": 20.0}


def round_tick(x: float, tick: float) -> float:
    return round(x / tick) * tick


def load_data(base_path: str = "/Users/orphee/Documents/Code_VSCODE/TRADOSMIC"):
    prices = pd.concat(
        [
            pd.read_csv(f"{base_path}/prices_round_0_day_-2.csv", sep=";"),
            pd.read_csv(f"{base_path}/prices_round_0_day_-1.csv", sep=";"),
        ],
        ignore_index=True,
    )
    trades_m2 = pd.read_csv(f"{base_path}/trades_round_0_day_-2.csv", sep=";")
    trades_m2["day"] = -2
    trades_m1 = pd.read_csv(f"{base_path}/trades_round_0_day_-1.csv", sep=";")
    trades_m1["day"] = -1
    trades = pd.concat([trades_m2, trades_m1], ignore_index=True).rename(
        columns={"symbol": "product"}
    )

    trade_groups = collections.defaultdict(list)
    for row in trades.itertuples(index=False):
        trade_groups[(int(row.day), int(row.timestamp), row.product)].append(
            (float(row.price), float(row.quantity))
        )

    prod_data = {}
    for prod in sorted(prices["product"].unique()):
        df = (
            prices[prices["product"] == prod]
            .sort_values(["day", "timestamp"])
            .reset_index(drop=True)
        )
        prod_data[prod] = list(
            zip(
                df["day"].astype(int),
                df["timestamp"].astype(int),
                df["bid_price_1"].astype(float),
                df["ask_price_1"].astype(float),
                df["mid_price"].astype(float),
            )
        )
    return prod_data, trade_groups


def simulate_policy(
    product: str,
    rows,
    trade_groups,
    depth_ticks: float,
    alpha: float,
    record: bool = False,
):
    tick = TICK_SIZES[product]
    limit = POSITION_LIMITS[product]
    pos, cash, prev_val = 0.0, 0.0, 0.0
    vals = np.empty(len(rows), dtype=float)
    incs = np.empty(len(rows), dtype=float)
    recs = [] if record else None

    for i, (day, ts, bb, ba, mid) in enumerate(rows):
        shift = -alpha * pos * tick
        bid = round_tick(bb - depth_ticks * tick + shift, tick)
        ask = round_tick(ba + depth_ticks * tick + shift, tick)
        buy_qty, sell_qty = 0.0, 0.0

        for price, qty in trade_groups.get((day, ts, product), ()):
            if price >= mid and ask <= price + 1e-9:
                if ask < ba - 1e-9:
                    fill = qty
                elif abs(ask - ba) < 1e-9:
                    fill = qty * QUEUE_SHARE_TOUCH
                else:
                    fill = qty
                fill = min(fill, limit + pos)
                if fill > 0:
                    sell_qty += fill
                    pos -= fill
                    cash += fill * ask

            if price <= mid and bid >= price - 1e-9:
                if bid > bb + 1e-9:
                    fill = qty
                elif abs(bid - bb) < 1e-9:
                    fill = qty * QUEUE_SHARE_TOUCH
                else:
                    fill = qty
                fill = min(fill, limit - pos)
                if fill > 0:
                    buy_qty += fill
                    pos += fill
                    cash -= fill * bid

        value = cash + pos * mid
        vals[i] = value
        incs[i] = value - prev_val
        prev_val = value

        if record:
            recs.append((
                day, ts, product,
                pos - buy_qty + sell_qty,
                bb, ba, bid, ask,
                buy_qty, sell_qty, pos, mid, cash, value, incs[i],
            ))

    std = incs.std()
    out = {
        "product": product,
        "depth_ticks": float(depth_ticks),
        "alpha": float(alpha),
        "final_pnl": float(vals[-1]),
        "mean_increment": float(incs.mean()),
        "vol_increment": float(std),
        "sharpe_empirical": float(incs.mean() / std) if std > 1e-12 else np.nan,
        "portfolio_value_vol": float(vals.std()),
        "inventory_final": float(pos),
    }

    if record:
        out["path"] = pd.DataFrame(
            recs,
            columns=[
                "day", "timestamp", "product", "position_before",
                "best_bid", "best_ask", "bid_quote", "ask_quote",
                "buy_qty", "sell_qty", "inventory_after",
                "mid_price", "cash", "portfolio_value", "pnl_increment",
            ],
        )
    return out


def _neg_sharpe(params, product, rows, trade_groups):
    """Objectif pour l'optimiseur : minimiser -Sharpe."""
    depth_ticks, alpha = params
    out = simulate_policy(product, rows, trade_groups, depth_ticks, alpha)
    s = out["sharpe_empirical"]
    return -s if np.isfinite(s) else 1e6


def optimize_product(
    product: str,
    rows,
    trade_groups,
    seed: int = 42,
    maxiter: int = 300,
    popsize: int = 12,
):
    """
    Differential Evolution sur (depth_ticks, alpha).

    Pourquoi DE plutôt que L-BFGS-B :
      - round_tick() introduit des discontinuités → gradient numérique faux
      - L-BFGS-B reste souvent coincé dans le minimum local le plus proche
      - DE explore l'espace globalement sans calculer de gradient
      - polish=True affine le meilleur candidat avec L-BFGS-B en fin de course,
        ce qui est safe car on part déjà d'un bon bassin d'attraction

    Bornes :
      depth_ticks ∈ [-2, 2]
      alpha       ∈ [0, 1]
    """
    res = differential_evolution(
        _neg_sharpe,
        bounds=[(-2.0, 2.0), (0.0, 1.0)],
        args=(product, rows, trade_groups),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,   # popsize × n_params individus dans la population
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,       # L-BFGS-B final pour affiner au sous-tick
        workers=1,
    )

    best_depth, best_alpha = res.x
    best_sharpe = -res.fun
    print(f"  [{product}] depth={best_depth:.4f}  alpha={best_alpha:.4f}  Sharpe={best_sharpe:.4f}")
    return best_depth, best_alpha, best_sharpe


def search_best_policies(
    base_path: str = "/Users/orphee/Documents/Code_VSCODE/TRADOSMIC",
    seed: int = 42,
    maxiter: int = 300,
    popsize: int = 12,
):
    prod_data, trade_groups = load_data(base_path)
    best = {}

    for product, rows in prod_data.items():
        print(f"[{product}] Optimisation...")
        depth, alpha, sharpe = optimize_product(
            product, rows, trade_groups,
            seed=seed, maxiter=maxiter, popsize=popsize,
        )
        best[product] = simulate_policy(
            product, rows, trade_groups, depth, alpha, record=True
        )

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    best["EMERALDS"]["path"].to_csv(
        f"{base_path}/best_position_policy_emeralds.csv", index=False
    )
    best["TOMATOES"]["path"].to_csv(
        f"{base_path}/best_position_policy_tomatoes.csv", index=False
    )

    combined = (
        best["EMERALDS"]["path"][["day", "timestamp", "portfolio_value", "pnl_increment"]]
        .copy()
        .rename(columns={
            "portfolio_value": "emeralds_value",
            "pnl_increment":   "emeralds_pnl_increment",
        })
    )
    combined["tomatoes_value"]         = best["TOMATOES"]["path"]["portfolio_value"].values
    combined["tomatoes_pnl_increment"] = best["TOMATOES"]["path"]["pnl_increment"].values
    combined["portfolio_value_total"]  = combined["emeralds_value"] + combined["tomatoes_value"]
    combined["pnl_increment_total"]    = combined["emeralds_pnl_increment"] + combined["tomatoes_pnl_increment"]
    combined.to_csv(f"{base_path}/best_combined_portfolio_path.csv", index=False)

    return best, combined


if __name__ == "__main__":
    BASE = "/Users/orphee/Documents/Code_VSCODE/TRADOSMIC"

    best, combined = search_best_policies(
        base_path=BASE,
        seed=42,
        maxiter=300,   # augmenter si beaucoup de données
        popsize=12,    # augmenter pour plus de robustesse (mais plus lent)
    )

    PARAMS = {
        product: {
            "alpha":       float(best[product]["alpha"]),
            "depth_ticks": float(best[product]["depth_ticks"]),
            "tick_size":   TICK_SIZES[product],
        }
        for product in best
    }

    with open(f"{BASE}/best_params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)

    print("\nParamètres sauvegardés :", PARAMS)
    print("\nPerformances :")
    for prod in best:
        r = best[prod]
        print(f"  {prod}: Sharpe={r['sharpe_empirical']:.4f}  PnL={r['final_pnl']:.1f}  inv_final={r['inventory_final']:.1f}")
# QUEUE_SHARE_TOUCH = 0
# TICK_SIZES = {"EMERALDS": 1, "TOMATOES": 1}
# POSITION_LIMITS = {"EMERALDS": 20.0, "TOMATOES": 20.0}

# def round_tick(x: float, tick: float) -> float:
#     return round(x / tick) * tick

# def load_data(base_path: str = "/Users/orphee/Documents/Code_VSCODE/TRADOSMIC"):
#     prices = pd.concat(
#         [
#             pd.read_csv(f"{base_path}/prices_round_0_day_-2.csv", sep=";"),
#             pd.read_csv(f"{base_path}/prices_round_0_day_-1.csv", sep=";"),
#         ],
#         ignore_index=True,
#     )
#     trades_m2 = pd.read_csv(f"{base_path}/trades_round_0_day_-2.csv", sep=";")
#     trades_m2["day"] = -2
#     trades_m1 = pd.read_csv(f"{base_path}/trades_round_0_day_-1.csv", sep=";")
#     trades_m1["day"] = -1
#     trades = pd.concat([trades_m2, trades_m1], ignore_index=True).rename(columns={"symbol": "product"})

#     trade_groups = collections.defaultdict(list)
#     for row in trades.itertuples(index=False):
#         trade_groups[(int(row.day), int(row.timestamp), row.product)].append(
#             (float(row.price), float(row.quantity))
#         )

#     prod_data = {}
#     for prod in sorted(prices["product"].unique()):
#         df = prices[prices["product"] == prod].sort_values(["day", "timestamp"]).reset_index(drop=True)
#         prod_data[prod] = list(
#             zip(
#                 df["day"].astype(int),
#                 df["timestamp"].astype(int),
#                 df["bid_price_1"].astype(float),
#                 df["ask_price_1"].astype(float),
#                 df["mid_price"].astype(float),
#             )
#         )
#     return prod_data, trade_groups

# def quote_from_position(best_bid: float, best_ask: float, position: float, tick: float, depth_ticks: int, alpha: float):
#     """
#     Politique fixe par position:
#       bid(q) = best_bid - depth_ticks*tick + round_tick(-alpha*q*tick)
#       ask(q) = best_ask + depth_ticks*tick + round_tick(-alpha*q*tick)

#     q > 0 (long)  -> on baisse bid et ask pour réduire le stock
#     q < 0 (short) -> on monte bid et ask pour racheter plus facilement
#     """
#     #shift = round_tick(-alpha * position * tick, tick)
#     shift = -alpha * position * tick
#     bid = round_tick(best_bid - depth_ticks * tick + shift, tick)
#     ask = round_tick(best_ask + depth_ticks * tick + shift, tick)
#     # if bid >= ask:
#     #     bid = ask - tick
#     return float(bid), float(ask)

# def simulate_policy(product: str, rows, trade_groups, depth_ticks: int, alpha: float, record: bool = False):
#     tick = TICK_SIZES[product]
#     limit = POSITION_LIMITS[product]
#     pos = 0.0
#     cash = 0.0
#     prev_val = 0.0
#     vals = np.empty(len(rows), dtype=float)
#     incs = np.empty(len(rows), dtype=float)
#     recs = [] if record else None

#     for i, (day, ts, bb, ba, mid) in enumerate(rows):
#         bid, ask = quote_from_position(bb, ba, pos, tick, depth_ticks, alpha)
#         buy_qty = 0.0
#         sell_qty = 0.0

#         for price, qty in trade_groups.get((day, ts, product), ()):
#             if price >= mid and ask <= price + 1e-9:
#                 if ask < ba - 1e-9:
#                     fill = qty
#                 elif abs(ask - ba) < 1e-9:
#                     fill = qty * QUEUE_SHARE_TOUCH
#                 else:
#                     fill = qty
#                 fill = min(fill, limit + pos)
#                 if fill > 0:
#                     sell_qty += fill
#                     pos -= fill
#                     cash += fill * ask

#             if price <= mid and bid >= price - 1e-9:
#                 if bid > bb + 1e-9:
#                     fill = qty
#                 elif abs(bid - bb) < 1e-9:
#                     fill = qty * QUEUE_SHARE_TOUCH
#                 else:
#                     fill = qty
#                 fill = min(fill, limit - pos)
#                 if fill > 0:
#                     buy_qty += fill
#                     pos += fill
#                     cash -= fill * bid

#         value = cash + pos * mid
#         vals[i] = value
#         incs[i] = value - prev_val
#         prev_val = value

#         if record:
#             recs.append(
#                 (
#                     day,
#                     ts,
#                     product,
#                     pos - buy_qty + sell_qty,  # position_before
#                     bb,
#                     ba,
#                     bid,
#                     ask,
#                     buy_qty,
#                     sell_qty,
#                     pos,
#                     mid,
#                     cash,
#                     value,
#                     incs[i],
#                 )
#             )

#     out = {
#         "product": product,
#         "depth_ticks": depth_ticks,
#         "alpha": alpha,
#         "final_pnl": float(vals[-1]),
#         "mean_increment": float(incs.mean()),
#         "vol_increment": float(incs.std()),
#         "sharpe_empirical": float(incs.mean() / incs.std()) if incs.std() > 1e-12 else np.nan,
#         "portfolio_value_vol": float(vals.std()),
#         "inventory_final": float(pos),
#     }

#     if record:
#         out["path"] = pd.DataFrame(
#             recs,
#             columns=[
#                 "day",
#                 "timestamp",
#                 "product",
#                 "position_before",
#                 "best_bid",
#                 "best_ask",
#                 "bid_quote",
#                 "ask_quote",
#                 "buy_qty",
#                 "sell_qty",
#                 "inventory_after",
#                 "mid_price",
#                 "cash",
#                 "portfolio_value",
#                 "pnl_increment",
#             ],
#         )
#     return out

# def search_best_policies(base_path: str = "/mnt/data"):
#     prod_data, trade_groups = load_data(base_path)
#     alphas = [round(x, 2) for x in np.linspace(0, 1.0, 51)]
#     depths = np.linspace(-2, 2,50)  # -2, -1, 0, 1, 2 ticks from the best bid/ask

#     summary = []
#     best = {}

#     for product, rows in prod_data.items():
#         best_score = -np.inf
#         best_policy = None
#         for depth_ticks, alpha in itertools.product(depths, alphas):
#             out = simulate_policy(product, rows, trade_groups, depth_ticks, alpha, record=False)
#             summary.append(out)
#             if np.isfinite(out["sharpe_empirical"]) and out["sharpe_empirical"] > best_score:
#                 best_score = out["sharpe_empirical"]
#                 best_policy = (depth_ticks, alpha)

#         best_depth, best_alpha = best_policy
#         best[product] = simulate_policy(product, rows, trade_groups, best_depth, best_alpha, record=True)

#     summary_df = pd.DataFrame(summary)
#     summary_df.to_csv(f"{base_path}/position_policy_search_summary.csv", index=False)
#     best["EMERALDS"]["path"].to_csv(f"{base_path}/best_position_policy_emeralds.csv", index=False)
#     best["TOMATOES"]["path"].to_csv(f"{base_path}/best_position_policy_tomatoes.csv", index=False)

#     combined = best["EMERALDS"]["path"][["day", "timestamp", "portfolio_value", "pnl_increment"]].copy()
#     combined = combined.rename(
#         columns={"portfolio_value": "emeralds_value", "pnl_increment": "emeralds_pnl_increment"}
#     )
#     combined["tomatoes_value"] = best["TOMATOES"]["path"]["portfolio_value"].values
#     combined["tomatoes_pnl_increment"] = best["TOMATOES"]["path"]["pnl_increment"].values
#     combined["portfolio_value_total"] = combined["emeralds_value"] + combined["tomatoes_value"]
#     combined["pnl_increment_total"] = combined["emeralds_pnl_increment"] + combined["tomatoes_pnl_increment"]
#     combined.to_csv(f"{base_path}/best_combined_portfolio_path.csv", index=False)

#     return summary_df, best, combined

# if __name__ == "__main__":
#     summary_df, best, combined = search_best_policies("/Users/orphee/Documents/Code_VSCODE/TRADOSMIC")
#     import json

#     PARAMS = {
#         product: {
#             "alpha": float(best[product]["alpha"]),
#             "depth_ticks": best[product]["depth_ticks"],
#             "tick_size": TICK_SIZES[product],
#         }
#         for product in best
#     }

#     with open("best_params.json", "w") as f:
#         json.dump(PARAMS, f)

#     print("Saved params:", PARAMS)
#     print(summary_df.sort_values(["product", "sharpe_empirical"], ascending=[True, False]).groupby("product").head(5))
#    #print("Best policies performance:", {product: {k: v for k, v in best[product].items() if k != "path"} for product in best})

