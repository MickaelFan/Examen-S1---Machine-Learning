import tkinter as tk
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────────────────
#  LOGIQUE DE JEU
# ─────────────────────────────────────────────────────────

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board):
    """Retourne 1 (X gagne), -1 (O gagne), 0 (nul), None (en cours)."""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0
    return None

def get_winning_line(board):
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return (a, b, c)
    return None

# ─────────────────────────────────────────────────────────
#  ÉVALUATION ML  (format p0..p8 — même que generator.py)
# ─────────────────────────────────────────────────────────

def encode_board_for_ml(board):
    """board[i] ∈ {1, -1, 0}  →  numpy array shape (1, 9)"""
    return np.array(board, dtype=float).reshape(1, 9)

def get_ml_score(board, m_wins, m_draw):
    """
    Score du point de vue de X : plus élevé = meilleur pour X.
    prob_wins * 1.0 + prob_draw * 0.5
    """
    feat = encode_board_for_ml(board)
    try:
        prob_w = m_wins.predict_proba(feat)[0][1]
        prob_d = m_draw.predict_proba(feat)[0][1]
        return prob_w * 1.0 + prob_d * 0.5
    except Exception:
        return 0.5

# ─────────────────────────────────────────────────────────
#  MINIMAX HYBRIDE (Alpha-Bêta + ML aux feuilles)
# ─────────────────────────────────────────────────────────

def minimax_hybrid(board, depth, is_max, alpha, beta, m_w, m_d):
    res = check_winner(board)
    if res == 1:  return 100
    if res == -1: return -100
    if res == 0:  return 0
    if depth == 0:
        return get_ml_score(board, m_w, m_d) * 10

    if is_max:
        val = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                val = max(val, minimax_hybrid(board, depth-1, False, alpha, beta, m_w, m_d))
                board[i] = 0
                alpha = max(alpha, val)
                if beta <= alpha: break
        return val
    else:
        val = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                val = min(val, minimax_hybrid(board, depth-1, True, alpha, beta, m_w, m_d))
                board[i] = 0
                beta = min(beta, val)
                if beta <= alpha: break
        return val

# ─────────────────────────────────────────────────────────
#  THÈME
# ─────────────────────────────────────────────────────────

THEME = {
    "bg":        "#0f1923",
    "panel":     "#1a2535",
    "panel2":    "#1e2d40",
    "border":    "#2d4060",
    "x_color":   "#38bdf8",
    "o_color":   "#f43f5e",
    "btn":       "#243044",
    "btn_hover": "#2d3f5a",
    "highlight": "#facc15",
    "green":     "#22c55e",
    "text_main": "#f1f5f9",
    "text_sub":  "#94a3b8",
    "text_warn": "#fb923c",
}

# ─────────────────────────────────────────────────────────
#  APPLICATION
# ─────────────────────────────────────────────────────────

class MorpionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISPM — Morpion IA · TEAM FIFA")
        self.root.geometry("460x730")
        self.root.resizable(False, False)
        self.root.configure(bg=THEME["bg"])

        # Chargement des modèles (cherche dans le dossier courant ET le dossier parent)
        self.m_w = self.m_d = None
        self.ia_ok = False
        for search_dir in ['.', '..', os.path.dirname(os.path.abspath(__file__))]:
            w_path = os.path.join(search_dir, 'model_wins.pkl')
            d_path = os.path.join(search_dir, 'model_draw.pkl')
            if os.path.exists(w_path) and os.path.exists(d_path):
                try:
                    self.m_w   = joblib.load(w_path)
                    self.m_d   = joblib.load(d_path)
                    self.ia_ok = True
                    break
                except Exception:
                    pass

        # État
        self.board     = [0] * 9
        self.turn      = 1          # 1 = X, -1 = O
        self.game_over = False
        self.scores    = {1: 0, -1: 0, 0: 0}

        # Mode par défaut : hybride si IA dispo, sinon HvH
        default_mode = 3 if self.ia_ok else 1
        self.mode = tk.IntVar(value=default_mode)

        self._build_ui()

        if not self.ia_ok:
            self._set_status(
                "Modeles non trouvés — lancez notebook.ipynb d'abord",
                THEME["text_warn"]
            )

    # ══════════════════════════════════════════════════════
    #  CONSTRUCTION UI
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        # En-tête
        hdr = tk.Frame(self.root, bg=THEME["bg"])
        hdr.pack(fill="x", padx=20, pady=(18, 4))
        tk.Label(hdr, text="ISPM · TEAM FIFA",
                 font=("Segoe UI", 10, "bold"),
                 bg=THEME["bg"], fg=THEME["text_sub"]).pack()
        tk.Label(hdr, text="MORPION IA",
                 font=("Segoe UI Black", 28, "bold"),
                 bg=THEME["bg"], fg=THEME["text_main"]).pack()

        self._build_scoreboard()
        self._build_mode_selector()

        # Statut
        self.status_var = tk.StringVar(value="AU TOUR DE X")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=("Segoe UI Black", 13),
            bg=THEME["bg"], fg=THEME["x_color"], pady=6
        )
        self.status_label.pack()

        self._build_grid()
        self._build_bottom_buttons()

    def _build_scoreboard(self):
        frame = tk.Frame(self.root, bg=THEME["panel"],
                         highlightbackground=THEME["border"],
                         highlightthickness=1)
        frame.pack(fill="x", padx=20, pady=(6, 4))

        self._score_vars = {}
        for col_idx, (lbl, color, key) in enumerate([
            ("X",   THEME["x_color"],  1),
            ("NUL", THEME["text_sub"], 0),
            ("O",   THEME["o_color"],  -1),
        ]):
            sub = tk.Frame(frame, bg=THEME["panel"])
            sub.grid(row=0, column=col_idx, padx=0, pady=10, sticky="nsew")
            frame.columnconfigure(col_idx, weight=1)

            tk.Label(sub, text=lbl, font=("Segoe UI", 10, "bold"),
                     bg=THEME["panel"], fg=color).pack()
            v = tk.StringVar(value="0")
            tk.Label(sub, textvariable=v, font=("Segoe UI Black", 22),
                     bg=THEME["panel"], fg=color).pack()
            self._score_vars[key] = v

    def _build_mode_selector(self):
        frame = tk.Frame(self.root, bg=THEME["panel2"],
                         highlightbackground=THEME["border"],
                         highlightthickness=1)
        frame.pack(fill="x", padx=20, pady=4)

        modes = [
            (1, "🧑 vs 🧑", "Humain vs Humain"),
            (2, "🧑 vs 🤖", "ML Pur (XGBoost)"),
            (3, "🧑 vs 🧠", "Hybride (Minimax + ML)"),
        ]
        for val, icon, label in modes:
            state = "normal"
            # Désactiver les modes IA si modèles absents
            if val in (2, 3) and not self.ia_ok:
                state = "disabled"
            tk.Radiobutton(
                frame, text=f"  {icon}  {label}",
                variable=self.mode, value=val, command=self.reset,
                font=("Segoe UI", 11, "bold"),
                bg=THEME["panel2"], fg=THEME["text_main"],
                selectcolor=THEME["border"],
                activebackground=THEME["panel2"],
                activeforeground=THEME["x_color"],
                cursor="hand2",
                state=state,
            ).pack(anchor="w", padx=16, pady=4)

    def _build_grid(self):
        outer = tk.Frame(self.root, bg=THEME["border"], padx=3, pady=3)
        outer.pack(padx=30, pady=6)
        inner = tk.Frame(outer, bg=THEME["border"])
        inner.pack()

        self.buttons = []
        for i in range(9):
            btn = tk.Button(
                inner, text="",
                font=("Segoe UI Black", 38), width=3, height=1,
                bg=THEME["btn"], activebackground=THEME["btn"],
                relief="flat", bd=0, cursor="hand2",
                command=lambda i=i: self.on_click(i),
            )
            btn.bind("<Enter>", lambda e, b=btn: self._hover_on(b))
            btn.bind("<Leave>", lambda e, b=btn: self._hover_off(b))
            btn.grid(row=i//3, column=i%3, padx=3, pady=3)
            self.buttons.append(btn)

    def _build_bottom_buttons(self):
        frame = tk.Frame(self.root, bg=THEME["bg"])
        frame.pack(pady=10)

        tk.Button(
            frame, text="  Nouvelle Partie  ",
            font=("Segoe UI Black", 12),
            bg=THEME["green"], fg="white",
            activebackground="#16a34a",
            relief="flat", cursor="hand2",
            padx=10, pady=8, command=self.reset,
        ).pack(side="left", padx=6)

        tk.Button(
            frame, text="  RAZ Scores  ",
            font=("Segoe UI", 11),
            bg=THEME["panel"], fg=THEME["text_sub"],
            activebackground=THEME["panel2"],
            relief="flat", cursor="hand2",
            padx=10, pady=8, command=self._reset_scores,
        ).pack(side="left", padx=6)

    # ══════════════════════════════════════════════════════
    #  HELPERS UI
    # ══════════════════════════════════════════════════════

    def _hover_on(self, btn):
        if btn["state"] == "normal":
            btn.config(bg=THEME["btn_hover"])

    def _hover_off(self, btn):
        if btn["state"] == "normal":
            btn.config(bg=THEME["btn"])

    def _set_status(self, text, color=None):
        self.status_var.set(text)
        if color:
            self.status_label.config(fg=color)

    def _update_scores(self):
        for key, var in self._score_vars.items():
            var.set(str(self.scores[key]))

    # ══════════════════════════════════════════════════════
    #  GAMEPLAY  ← correctif principal
    # ══════════════════════════════════════════════════════

    def on_click(self, i):
        """Gère un clic sur la case i."""
        if self.game_over or self.board[i] != 0:
            return

        mode = self.mode.get()

        # ── Mode IA : seul X (turn == 1) peut cliquer ──
        if mode in (2, 3) and self.turn != 1:
            return

        # Jouer le coup du joueur courant
        self._place(i, self.turn)
        if self.game_over:
            return

        if mode == 1:
            # ── Humain vs Humain : alterner simplement ──
            self.turn *= -1
            self._refresh_turn_label()

        else:
            # ── Mode IA : déclencher l'IA après un délai ──
            self.turn = -1
            self._set_status("L'IA réfléchit…", THEME["text_warn"])
            self.root.after(400, self._ia_move)

    def _place(self, i, player):
        """Place le symbole du joueur sur la case i et vérifie la fin de partie."""
        self.board[i] = player
        color = THEME["x_color"] if player == 1 else THEME["o_color"]
        char  = "X" if player == 1 else "O"
        self.buttons[i].config(
            text=char, state="disabled",
            disabledforeground=color, bg=THEME["panel"]
        )
        result = check_winner(self.board)
        if result is not None:
            self._end_game(result)

    def _ia_move(self):
        """Calcule et joue le coup de l'IA (O = -1)."""
        if not self.ia_ok:
            self._set_status("Modèles manquants !", THEME["text_warn"])
            return

        move = -1
        mode = self.mode.get()

        if mode == 2:
            # ── ML PUR : choisir la case qui minimise le score pour X ──
            best = float('inf')
            for i in range(9):
                if self.board[i] == 0:
                    self.board[i] = -1
                    s = get_ml_score(self.board, self.m_w, self.m_d)
                    self.board[i] = 0
                    if s < best:
                        best = s
                        move = i
        else:
            # ── HYBRIDE : Minimax profondeur 3 + ML aux feuilles ──
            best = float('inf')
            for i in range(9):
                if self.board[i] == 0:
                    self.board[i] = -1
                    s = minimax_hybrid(
                        self.board, 3, True,
                        -float('inf'), float('inf'),
                        self.m_w, self.m_d
                    )
                    self.board[i] = 0
                    if s < best:
                        best = s
                        move = i

        if move != -1:
            self._place(move, -1)

        if not self.game_over:
            self.turn = 1
            self._refresh_turn_label()

    def _refresh_turn_label(self):
        if self.turn == 1:
            self._set_status("AU TOUR DE X", THEME["x_color"])
        else:
            self._set_status("AU TOUR DE O", THEME["o_color"])

    def _end_game(self, result):
        self.game_over = True
        self.scores[result] = self.scores.get(result, 0) + 1
        self._update_scores()

        # Surligner la ligne gagnante
        line = get_winning_line(self.board)
        if line:
            for idx in line:
                self.buttons[idx].config(bg=THEME["highlight"])

        # Désactiver toutes les cases
        for btn in self.buttons:
            btn.config(state="disabled")

        if result == 1:
            self._set_status("VICTOIRE DE X !", THEME["x_color"])
            msg, color = "X  REMPORTE\nLA PARTIE !", THEME["x_color"]
        elif result == -1:
            self._set_status("VICTOIRE DE O !", THEME["o_color"])
            msg, color = "O  REMPORTE\nLA PARTIE !", THEME["o_color"]
        else:
            self._set_status("MATCH NUL !", THEME["text_sub"])
            msg, color = "MATCH NUL !", THEME["text_sub"]

        self.root.after(700, lambda: self._show_popup(msg, color))

    def _show_popup(self, msg, color):
        popup = tk.Toplevel(self.root)
        popup.title("Résultat")
        popup.geometry("320x180")
        popup.configure(bg=THEME["panel"])
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()
        popup.geometry(f"+{self.root.winfo_x()+70}+{self.root.winfo_y()+270}")

        tk.Frame(popup, bg=color, height=4).pack(fill="x")
        tk.Label(popup, text=msg,
                 font=("Segoe UI Black", 17),
                 bg=THEME["panel"], fg=color,
                 justify="center").pack(pady=26)
        tk.Button(popup, text="  Nouvelle Partie  ",
                  font=("Segoe UI Black", 11),
                  bg=THEME["green"], fg="white",
                  activebackground="#16a34a",
                  relief="flat", cursor="hand2",
                  padx=8, pady=6,
                  command=lambda: [popup.destroy(), self.reset()]
                  ).pack()

    def reset(self):
        """Remet le plateau à zéro pour une nouvelle partie."""
        self.board     = [0] * 9
        self.turn      = 1
        self.game_over = False
        for btn in self.buttons:
            btn.config(text="", state="normal", bg=THEME["btn"],
                       disabledforeground=THEME["text_main"])
        self._refresh_turn_label()

    def _reset_scores(self):
        self.scores = {1: 0, -1: 0, 0: 0}
        self._update_scores()


# ─────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = MorpionApp(root)
    root.mainloop()