from __future__ import annotations

from collections import defaultdict
from typing import Callable

from hoola.action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from hoola.card import Card, Suit
from hoola.logger import format_cards
from hoola.observation import Observation


SUIT_NAMES = {
    Suit.CLUBS: "Clubs    ♣",
    Suit.DIAMONDS: "Diamonds ♦",
    Suit.HEARTS: "Hearts   ♥",
    Suit.SPADES: "Spades   ♠",
}


class HumanAgent:
    """Interactive CLI agent for human-vs-AI play."""

    def __init__(self, input_func: Callable[[str], str] | None = None) -> None:
        self.input_func = input_func or input

    def select_action(self, observation: Observation, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions available")

        ordered = self._ordered_actions(legal_actions)
        while True:
            self._print_position(observation)
            self._print_actions(ordered, observation)

            raw = self.input_func("\nYour choice > ").strip().lower()
            if raw in {"h", "help", "?", "b", "board", ""}:
                continue
            if raw in {"q", "quit", "exit"}:
                raise KeyboardInterrupt("Human player quit the game")

            try:
                choice = int(raw)
            except ValueError:
                print("Invalid input: enter an action number, 'h' for help/board, or 'q' to quit.")
                continue

            if 1 <= choice <= len(ordered):
                selected = ordered[choice - 1]
                print(f"You chose: {self._format_action(selected, observation)}")
                return selected

            print(f"Choice out of range: enter 1-{len(ordered)}.")

    @staticmethod
    def _ordered_actions(legal_actions: list[Action]) -> list[Action]:
        def key(action: Action):
            if isinstance(action, DrawStockAction):
                return (0, 0, "")
            if isinstance(action, ThankYouAction):
                return (1, -len(action.meld_cards), format_cards(action.meld_cards))
            if isinstance(action, MeldAction):
                return (2, -len(action.cards), format_cards(action.cards))
            if isinstance(action, LayoffAction):
                return (3, -len(action.cards), f"{action.meld_id:04d}")
            if isinstance(action, PassPlayAction):
                return (4, 0, "")
            if isinstance(action, DiscardAction):
                return (5, action.card.id, "")
            return (99, 0, repr(action))

        return sorted(legal_actions, key=key)

    @staticmethod
    def _category(action: Action) -> str:
        if isinstance(action, DrawStockAction):
            return "DRAW"
        if isinstance(action, ThankYouAction):
            return "THANK YOU"
        if isinstance(action, MeldAction):
            return "MELD"
        if isinstance(action, LayoffAction):
            return "LAYOFF"
        if isinstance(action, PassPlayAction):
            return "PASS"
        if isinstance(action, DiscardAction):
            return "DISCARD"
        return "OTHER"

    def _print_actions(self, actions: list[Action], observation: Observation) -> None:
        print("\nLEGAL ACTIONS")
        print("-" * 72)
        last_category = None
        for idx, action in enumerate(actions, start=1):
            category = self._category(action)
            if category != last_category:
                if last_category is not None:
                    print()
                print(f"[{category}]")
                last_category = category
            print(f"  {idx:2d}) {self._format_action(action, observation)}")
        print("\nCommands: number=choose | h/?=show board again | q=quit")

    def _print_position(self, observation: Observation) -> None:
        print("\n" + "=" * 72)
        print(
            f"YOUR TURN | Player {observation.player_id} | "
            f"phase={observation.phase.name} | turn={observation.turns_completed + 1}"
        )
        print("=" * 72)
        self._print_hand(observation.my_hand)

        top = str(observation.discard_pile[-1]) if observation.discard_pile else "(none)"
        print("\nPUBLIC INFO")
        print(f"  Opponent hand : {observation.opponent_hand_size} cards")
        print(f"  Stock         : {observation.stock_size} cards")
        print(f"  Top discard   : {top}")
        you = observation.player_id
        opp = 1 - you
        print(
            f"  Registered    : YOU={'yes' if observation.has_registered[you] else 'no'} | "
            f"OPP={'yes' if observation.has_registered[opp] else 'no'}"
        )

        print("\nTABLE MELDS")
        if observation.melds:
            for i, meld in enumerate(observation.melds):
                owner = "YOU" if meld.owner == observation.player_id else "OPP"
                print(f"  #{i:<2d} {owner:<3} : {format_cards(meld.cards)}")
        else:
            print("  (none)")

    @staticmethod
    def _print_hand(cards: tuple[Card, ...]) -> None:
        grouped: dict[Suit, list[Card]] = defaultdict(list)
        for card in sorted(cards, key=lambda c: c.id):
            grouped[card.suit].append(card)

        print(f"YOUR HAND ({len(cards)} cards)")
        for suit in Suit:
            suit_cards = grouped.get(suit, [])
            text = " ".join(str(card) for card in suit_cards) if suit_cards else "-"
            print(f"  {SUIT_NAMES[suit]} : {text}")

    def _format_action(self, action: Action, observation: Observation) -> str:
        if isinstance(action, DrawStockAction):
            return "Draw one card from stock"
        if isinstance(action, ThankYouAction):
            taken = observation.discard_pile[-1] if observation.discard_pile else "?"
            return f"THANK YOU: take {taken} and register {format_cards(action.meld_cards)}"
        if isinstance(action, MeldAction):
            if observation.phase.name == "DRAW":
                return f"Register {format_cards(action.cards)} WITHOUT drawing"
            return f"Register {format_cards(action.cards)}"
        if isinstance(action, LayoffAction):
            target = observation.melds[action.meld_id]
            owner = "YOU" if target.owner == observation.player_id else "OPP"
            suffix = " WITHOUT drawing" if observation.phase.name == "DRAW" else ""
            return (
                f"Lay off {format_cards(action.cards)} -> Meld #{action.meld_id} "
                f"({owner}: {format_cards(target.cards)}){suffix}"
            )
        if isinstance(action, PassPlayAction):
            return "Finish PLAY phase and proceed to discard"
        if isinstance(action, DiscardAction):
            return f"Discard {action.card}"
        return repr(action)
