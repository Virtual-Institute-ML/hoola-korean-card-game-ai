from __future__ import annotations

from dataclasses import dataclass

from .action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from .card import Card
from .state import GameState


def format_cards(cards) -> str:
    cards = list(cards)
    if not cards:
        return "(empty)"
    return " ".join(str(card) for card in sorted(cards, key=lambda c: c.id))


@dataclass(frozen=True, slots=True)
class ActionSnapshot:
    player: int
    hand_before: tuple[Card, ...]
    stock_size_before: int
    discard_top_before: Card | None


class ConsoleGameLogger:
    """Human-readable console logger with optional hidden-hand protection."""

    def __init__(
        self,
        player_names: list[str] | tuple[str, str] | None = None,
        hidden_hand_players: set[int] | None = None,
    ) -> None:
        self._last_header_turn: tuple[int, int] | None = None
        self.player_names = tuple(player_names) if player_names is not None else ("Player 0", "Player 1")
        if len(self.player_names) != 2:
            raise ValueError("player_names must contain exactly two labels")
        self.hidden_hand_players = set(hidden_hand_players or set())

    @staticmethod
    def snapshot(state: GameState) -> ActionSnapshot:
        player = state.current_player
        return ActionSnapshot(
            player=player,
            hand_before=tuple(state.hands[player]),
            stock_size_before=len(state.stock),
            discard_top_before=state.discard_pile[-1] if state.discard_pile else None,
        )

    def begin_turn(self, state: GameState) -> None:
        key = (state.turns_completed, state.current_player)
        if key == self._last_header_turn:
            return
        self._last_header_turn = key

        player = state.current_player
        top = str(state.discard_pile[-1]) if state.discard_pile else "(none)"
        print("=" * 72)
        print(
            f"Turn {state.turns_completed + 1} | {self.player_names[player]} | "
            f"hand={len(state.hands[player])} | stock={len(state.stock)} | "
            f"top discard={top}"
        )
        if player in self.hidden_hand_players:
            print("  HAND       : [hidden]")
        else:
            print(f"  HAND       : {format_cards(state.hands[player])}")

    def log_action(self, snap: ActionSnapshot, action: Action, state: GameState) -> None:
        player = snap.player
        hidden = player in self.hidden_hand_players
        hand_after = tuple(state.hands[player])
        before_n = len(snap.hand_before)
        after_n = len(hand_after)
        count = f"hand {before_n} -> {after_n}"

        if isinstance(action, DrawStockAction):
            if hidden:
                drawn_text = "[hidden card]"
            else:
                drawn = [card for card in hand_after if card not in snap.hand_before]
                drawn_text = str(drawn[0]) if len(drawn) == 1 else "?"
            print(f"  DRAW       : {drawn_text:<20} {count}")

        elif isinstance(action, ThankYouAction):
            taken = snap.discard_top_before
            print(
                f"  THANK YOU! : take {taken} -> meld "
                f"{format_cards(action.meld_cards):<17} {count}"
            )

        elif isinstance(action, MeldAction):
            print(f"  MELD       : {format_cards(action.cards):<20} {count}")

        elif isinstance(action, LayoffAction):
            print(
                f"  LAYOFF     : {format_cards(action.cards)} -> Meld #{action.meld_id:<8} "
                f"{count}"
            )

        elif isinstance(action, PassPlayAction):
            print("  PASS       : finish PLAY phase")

        elif isinstance(action, DiscardAction):
            print(f"  DISCARD    : {action.card!s:<20} {count}")

        if state.terminal and state.winner == player:
            print(f"  >>> {self.player_names[player]} WINS immediately: hand is empty.")
        elif isinstance(action, DiscardAction):
            if hidden:
                print(f"  HAND LEFT  : [hidden] ({after_n} cards)")
            else:
                print(f"  HAND LEFT  : {format_cards(hand_after)}")

    def finish(self, state: GameState, action_count: int) -> None:
        print("=" * 72)
        if state.is_draw:
            print(f"GAME OVER | draw | turns={state.turns_completed} | actions={action_count}")
        else:
            print(
                f"GAME OVER | winner={self.player_names[state.winner]} | "
                f"turns={state.turns_completed} | actions={action_count}"
            )
