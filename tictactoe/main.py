import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
console.print(f"[bold green]Using device: {device}[/]")


# -------------------------
# Environment
# -------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 1.0  # 1 = X (starts), -1 = O
        return self.board.copy()

    def available_actions(self):
        return list(np.where(self.board == 0)[0])

    def step(self, action):
        if self.board[action] != 0:
            return self.board.copy(), -10, True  # Illegal move

        self.board[action] = self.current_player

        winner = self.check_winner()
        done = winner != 0 or len(self.available_actions()) == 0
        reward = (
            1.0 if winner == self.current_player else 0.0
        )  # +1 if the player who moved won

        self.current_player *= -1
        return self.board.copy(), reward, done

    def check_winner(self):
        wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]  # +1 X, -1 O
        return 0

    def render(self):
        sym = {1.0: "[bold red]X[/]", -1.0: "[bold blue]O[/]", 0.0: " "}
        table = Table.grid(padding=1)
        for i in range(3):
            table.add_row(
                sym[self.board[i * 3]],
                sym[self.board[i * 3 + 1]],
                sym[self.board[i * 3 + 2]],
            )
        console.print(table)


# -------------------------
# Policy Network
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 9)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=-1)


# -------------------------
# Self-Play Training
# -------------------------
def train_self_play(epochs=50000, report_every=5000):
    model = PolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    gamma = 0.95

    x_wins = o_wins = draws = 0

    console.print(
        Panel(
            "[bold magenta]Starting long self-play training – this will make the AI unbeatable![/]"
        )
    )

    for epoch in track(range(epochs), description="Self-Play Training"):
        env = TicTacToeEnv()
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            probs = model(state_t)[0]

            avail = env.available_actions()
            if not avail:
                break

            # Mask invalid actions
            mask = torch.full((9,), -float("inf"), device=device)
            mask[avail] = 0
            masked_logits = probs.log() + mask
            action_dist = torch.softmax(masked_logits, dim=-1)
            action = torch.multinomial(action_dist, 1).item()

            log_prob = torch.log(probs[action] + 1e-8)
            log_probs.append(log_prob)

            state, reward, done = env.step(action)
            rewards.append(reward)

        if rewards:  # Game finished
            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, device=device)

            # Baseline
            baseline = returns.mean()
            advantages = returns - baseline

            # REINFORCE loss
            loss = -(torch.stack(log_probs) * advantages).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats (who won from starting perspective)
            winner = env.check_winner()
            if winner == 1.0:
                x_wins += 1
            elif winner == -1.0:
                o_wins += 1
            else:
                draws += 1

        # Report
        if (epoch + 1) % report_every == 0:
            total = x_wins + o_wins + draws
            if total > 0:
                console.print(f"\n[bold cyan]After {epoch + 1} games:[/]")
                console.print(
                    f"X wins: {x_wins / total * 100:.1f}% | O wins: {o_wins / total * 100:.1f}% | Draws: {draws / total * 100:.1f}%"
                )
            x_wins = o_wins = draws = 0

    return model


# -------------------------
# Play Against AI (You = X, AI = O)
# -------------------------
def play_vs_ai(model):
    model.eval()
    env = TicTacToeEnv()
    state = env.reset()

    console.print(
        Panel(
            "[bold green]You are X (first move). AI is O. Good luck – it's tough now![/]"
        )
    )
    console.print("[yellow]Positions:\n1 2 3\n4 5 6\n7 8 9[/]")

    done = False
    while not done:
        env.render()

        if env.current_player == 1.0:  # Human (X)
            while True:
                try:
                    move = int(input("\n[bold green]Your move (1-9): [/]")) - 1
                    if move in env.available_actions():
                        state, _, done = env.step(move)
                        break
                    console.print("[red]Invalid move – spot taken![/]")
                except:
                    console.print("[red]Enter a number 1-9[/]")
        else:  # AI (O)
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                probs = model(state_t)[0]
                avail = env.available_actions()
                mask = torch.full((9,), -float("inf"))
                mask[avail] = 0
                action = torch.argmax(probs + mask).item()
            state, _, done = env.step(action)
            console.print(f"[bold blue]AI plays position {action + 1}[/]")

        if done:
            env.render()
            winner = env.check_winner()
            if winner == 1.0:
                console.print(Panel("[bold green]You win! 🎉[/]"))
            elif winner == -1.0:
                console.print(Panel("[bold red]AI wins! 🤖[/]"))
            else:
                console.print(Panel("[bold white]Draw! 👏[/]"))
            break


# -------------------------
# RUN
# -------------------------
model = train_self_play(epochs=50000, report_every=5000)
play_vs_ai(model)
