import tkinter as tk
import numpy as np
import random

class Morpion:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Morpion Amélioré")
        self.difficulty = "Difficile"  # Niveau de difficulté par défaut
        self.multiplayer = False  # Mode multijoueur désactivé par défaut
        self.cache = {}  # Cache pour minimax
        self.create_menu()
        self.reset_game()
        self.window.mainloop()

    def create_menu(self):
        """Créer un menu pour sélectionner la difficulté et le mode."""
        menu = tk.Menu(self.window)
        self.window.config(menu=menu)

        # Menu Niveau de Difficulté
        difficulty_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Difficulté", menu=difficulty_menu)
        difficulty_menu.add_command(label="Facile", command=lambda: self.set_difficulty("Facile"))
        difficulty_menu.add_command(label="Moyen", command=lambda: self.set_difficulty("Moyen"))
        difficulty_menu.add_command(label="Difficile", command=lambda: self.set_difficulty("Difficile"))

        # Menu Mode
        mode_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Mode", menu=mode_menu)
        mode_menu.add_command(label="Solo (avec IA)", command=lambda: self.set_multiplayer(False))
        mode_menu.add_command(label="Multijoueur", command=lambda: self.set_multiplayer(True))

    def set_difficulty(self, difficulty):
        """Changer le niveau de difficulté."""
        self.difficulty = difficulty
        self.reset_game()

    def set_multiplayer(self, multiplayer):
        """Activer ou désactiver le mode multijoueur."""
        self.multiplayer = multiplayer
        self.reset_game()

    def reset_game(self):
        """Réinitialiser la grille et l'interface."""
        self.board = np.zeros((3, 3), dtype=int)  # Grille de jeu
        self.buttons = [[None for _ in range(3)] for _ in range(3)]  # Boutons pour l'interface
        self.first_player = random.choice(["player", "ai"])  # Choisir aléatoirement qui commence
        self.current_player = self.first_player  # Définir le joueur courant
        self.cache.clear()  # Réinitialiser le cache
        for widget in self.window.winfo_children():
            if not isinstance(widget, tk.Menu):  # Ne pas supprimer le menu
                widget.destroy()
        self.create_board()
        if self.first_player == "ai" and not self.multiplayer:
            self.ai_move()

    def create_board(self):
        """Créer la grille de boutons pour le jeu."""
        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    self.window,
                    text="",
                    font=("Helvetica", 24),
                    height=2,
                    width=5,
                    bg="lightblue",
                    command=lambda row=i, col=j: self.player_move(row, col)
                )
                button.grid(row=i, column=j)
                self.buttons[i][j] = button
        self.message_label = tk.Label(
            self.window,
            text=f"{'IA' if self.first_player == 'ai' else 'Vous'} commencez !",
            font=("Helvetica", 16)
        )
        self.message_label.grid(row=3, column=0, columnspan=3)
        self.restart_button = tk.Button(
            self.window,
            text="Recommencer",
            font=("Helvetica", 12),
            command=self.reset_game
        )
        self.restart_button.grid(row=4, column=0, columnspan=3)

    def player_move(self, row, col):
        """Action du joueur."""
        if self.board[row, col] == 0 and (self.current_player == "player" or self.multiplayer):
            self.board[row, col] = -1 if self.current_player == "player" else 1
            self.update_buttons()
            if self.check_winner(self.board[row, col]):
                self.end_game(f"{'Vous' if self.current_player == 'player' else 'Joueur 2'} avez gagné !")
            elif self.is_draw():
                self.end_game("Match nul !")
            else:
                self.current_player = "ai" if self.current_player == "player" and not self.multiplayer else "player"
                if self.current_player == "ai":
                    self.ai_move()

    def ai_move(self):
        """Action de l'IA."""
        if self.difficulty == "Facile":
            self.random_move()
        elif self.difficulty == "Moyen":
            if random.random() < 0.5:
                self.random_move()
            else:
                self.optimal_move()
        elif self.difficulty == "Difficile":
            self.optimal_move()
        self.update_buttons()
        if self.check_winner(1):
            self.end_game("L'IA a gagné !")
        elif self.is_draw():
            self.end_game("Match nul !")
        else:
            self.current_player = "player"

    def random_move(self):
        """Effectuer un coup aléatoire."""
        empty_cells = np.argwhere(self.board == 0)
        if empty_cells.size > 0:
            i, j = empty_cells[np.random.choice(len(empty_cells))]
            self.board[i, j] = 1

    def optimal_move(self):
        """Effectuer le meilleur coup possible."""
        # Vérifier si l'IA peut gagner ou bloquer immédiatement
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    self.board[i, j] = 1
                    if self.check_winner(1):
                        return  # Coup gagnant pour l'IA
                    self.board[i, j] = 0
                    self.board[i, j] = -1
                    if self.check_winner(-1):
                        self.board[i, j] = 1  # Bloquer le joueur
                        return
                    self.board[i, j] = 0

        # Sinon, utiliser Minimax pour trouver le meilleur coup
        best_score = -float("inf")
        best_move = None
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    self.board[i, j] = 1
                    score = self.minimax(self.board, False, -float("inf"), float("inf"), 4)
                    self.board[i, j] = 0
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        if best_move:
            self.board[best_move[0], best_move[1]] = 1

    def minimax(self, board, is_maximizing, alpha, beta, depth):
        """Algorithme Minimax avec élagage alpha-bêta et limite de profondeur."""
        if depth == 0 or self.check_winner(1) or self.check_winner(-1) or self.is_draw():
            return self.evaluate_board(board)

        if is_maximizing:
            best_score = -float("inf")
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = 1
                        score = self.minimax(board, False, alpha, beta, depth - 1)
                        board[i, j] = 0
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
        else:
            best_score = float("inf")
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = -1
                        score = self.minimax(board, True, alpha, beta, depth - 1)
                        board[i, j] = 0
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
        return best_score

    def evaluate_board(self, board):
        """Évaluer l'état de la grille."""
        if self.check_winner(1):
            return 10  # Victoire IA
        if self.check_winner(-1):
            return -10  # Victoire joueur

        # Ajouter des points pour des configurations favorables
        score = 0
        for i in range(3):
            score += self.evaluate_line(board[i, :])  # Lignes
            score += self.evaluate_line(board[:, i])  # Colonnes
        score += self.evaluate_line(board.diagonal())  # Diagonale principale
        score += self.evaluate_line(np.fliplr(board).diagonal())  # Diagonale secondaire
        return score

    def evaluate_line(self, line):
        """Évaluer une ligne, colonne ou diagonale."""
        if np.count_nonzero(line == 1) == 2 and np.count_nonzero(line == 0) == 1:
            return 5  # L'IA est proche de gagner
        elif np.count_nonzero(line == -1) == 2 and np.count_nonzero(line == 0) == 1:
            return -5  # Le joueur est proche de gagner
        elif np.count_nonzero(line == 1) == 1 and np.count_nonzero(line == 0) == 2:
            return 2  # Opportunité pour l'IA
        elif np.count_nonzero(line == -1) == 1 and np.count_nonzero(line == 0) == 2:
            return -2  # Opportunité pour le joueur
        return 0  # Ligne neutre

    def update_buttons(self):
        """Mettre à jour l'affichage des boutons en fonction de la grille."""
        symbols = {0: "", 1: "O", -1: "X"}
        colors = {0: "lightblue", 1: "lightgreen", -1: "lightcoral"}
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]["text"] = symbols[self.board[i, j]]
                self.buttons[i][j]["bg"] = colors[self.board[i, j]]

    def check_winner(self, player):
        """Vérifier si un joueur a gagné."""
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all(self.board.diagonal() == player) or all(np.fliplr(self.board).diagonal() == player):
            return True
        return False

    def is_draw(self):
        """Vérifier si la grille est pleine (match nul)."""
        return not (self.board == 0).any()

    def end_game(self, message):
        """Terminer le jeu et afficher un message."""
        for row in self.buttons:
            for button in row:
                button.config(state=tk.DISABLED)  # Désactiver tous les boutons
        self.message_label.config(text=message)

# Lancer le jeu
Morpion()
