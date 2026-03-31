import pandas as pd
import random
import os

# ─────────────────────────────────────────────────────────
#  LOGIQUE DE VICTOIRE
# ─────────────────────────────────────────────────────────

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),   # lignes
    (0,3,6),(1,4,7),(2,5,8),   # colonnes
    (0,4,8),(2,4,6)            # diagonales
]

def check_winner(board):
    """Retourne 1 (X gagne), -1 (O gagne), 0 (nul), None (en cours)."""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0
    return None

# ─────────────────────────────────────────────────────────
#  SIMULATION D'UNE PARTIE COMPLÈTE
# ─────────────────────────────────────────────────────────

def simulate_full_game():
    """
    Simule une partie entière jusqu'à la fin (victoire ou nul).
    Retourne (board_final, résultat).
    On capture TOUTES les positions intermédiaires après chaque coup.
    """
    board = [0] * 9
    current = 1  # X commence toujours
    snapshots = []

    while True:
        empty = [i for i, x in enumerate(board) if x == 0]
        if not empty:
            break
        idx = random.choice(empty)
        board[idx] = current

        result = check_winner(board)
        if result is not None:
            # On enregistre la position finale
            snapshots.append((list(board), result))
            break

        # Position intermédiaire : on note aussi (pour enrichir le dataset)
        # mais uniquement si au moins 4 coups ont été joués (plateau plus informatif)
        if board.count(0) <= 5:
            snapshots.append((list(board), None))  # résultat encore inconnu

        current *= -1

    return snapshots, result if result is not None else 0


def generate_dataset(n_games=10000):
    """
    Génère n_games parties et collecte les positions finales terminées.
    Seules les positions où la partie EST finie sont retenues.
    """
    data = []
    games_done = 0

    while games_done < n_games:
        board = [0] * 9
        current = 1

        # Jouer une partie complète
        while True:
            empty = [i for i, x in enumerate(board) if x == 0]
            if not empty:
                break
            board[random.choice(empty)] = current
            result = check_winner(board)
            if result is not None:
                break
            current *= -1

        result = check_winner(board)
        if result is None:
            result = 0  # nul (plateau plein sans gagnant détecté)

        x_wins  = 1 if result == 1  else 0
        is_draw = 1 if result == 0  else 0

        data.append(board + [x_wins, is_draw])
        games_done += 1

    cols = [f'p{i}' for i in range(9)] + ['x_wins', 'is_draw']
    return pd.DataFrame(data, columns=cols)


# ─────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────

def main():
    # Chemin relatif depuis la racine du projet (D:\ISPM\M1\Examen\)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'ressources')
    output_path = os.path.join(output_dir, 'dataset.csv')

    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Génération de 10 000 parties pour TEAM FIFA...")
    df = generate_dataset(n_games=10000)

    df.to_csv(output_path, index=False)

    total   = len(df)
    x_wins  = df['x_wins'].sum()
    draws   = df['is_draw'].sum()
    o_wins  = total - x_wins - draws

    print(f"✅ '{output_path}' créé avec succès !")
    print(f"   Total lignes : {total}")
    print(f"   X gagne      : {x_wins}  ({x_wins/total*100:.1f} %)")
    print(f"   O gagne      : {o_wins}  ({o_wins/total*100:.1f} %)")
    print(f"   Match nul    : {draws}  ({draws/total*100:.1f} %)")


if __name__ == "__main__":
    main()