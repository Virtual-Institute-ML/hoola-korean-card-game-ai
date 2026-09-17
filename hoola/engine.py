from __future__ import annotations

import random
from dataclasses import dataclass
import copy
from itertools import combinations

from .action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from .card import Card, create_deck
from .meld import is_valid_layoff_result, find_valid_melds
from .rules import DEFAULT_RULES, RuleConfig, valid_thank_you_melds
from .state import GameState, TableMeld, TurnPhase


class IllegalAction(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GameResult:
    winner: int | None
    is_draw: bool
    turns_completed: int


class HoolaEngine:
    def __init__(
        self,
        rules: RuleConfig = DEFAULT_RULES,
        seed: int | None = None,
        max_turns: int = 500,
    ) -> None:
        if rules.num_players != 2:
            raise ValueError("This engine currently supports exactly 2 players")
        self.rules = rules
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.state = self._new_game()

    def _new_game(self) -> GameState:
        deck = create_deck()
        self.rng.shuffle(deck)

        hands = [[], []]
        for _ in range(self.rules.starting_hand_size):
            for player in (0, 1):
                hands[player].append(deck.pop())

        # Initial face-up discard is public but was not discarded by either player,
        # so it cannot trigger Thank You on turn 1.
        discard_pile = [deck.pop()]

        return GameState(
            hands=hands,
            stock=deck,
            discard_pile=discard_pile,
            has_registered=[False, False],
            current_player=0,
            phase=TurnPhase.DRAW,
        )

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        self.state = self._new_game()
        return self.state

    def clone_state(self) -> GameState:
        return copy.deepcopy(self.state)

    def _canonical_cards(self, cards) -> tuple[Card, ...]:
        return tuple(sorted(cards, key=lambda c: c.id))

    def _can_draw_stock(self) -> bool:
        return bool(self.state.stock) or len(self.state.discard_pile) > 1

    def _recycle_stock_if_needed(self) -> None:
        if self.state.stock:
            return
        if len(self.state.discard_pile) <= 1:
            return

        top = self.state.discard_pile[-1]
        recyclable = self.state.discard_pile[:-1]
        self.rng.shuffle(recyclable)
        self.state.stock = recyclable
        self.state.discard_pile = [top]

    def _layoff_actions(self, player: int, hand: list[Card]) -> list[LayoffAction]:
        """Enumerate normalized atomic layoff actions.

        Every strategic multi-card layoff can be expressed as a sequence of
        these actions.  Single-card extensions cover normal runs/sets and the
        6/8 extension of a singleton seven.  The only necessary two-card atom
        is adding two sevens to a singleton seven to create a 3-card set;
        neither seven can be added alone because a two-card same-rank group is
        not a legal table state.
        """
        if not self.state.has_registered[player]:
            return []

        actions: list[LayoffAction] = []
        for meld_id, table_meld in enumerate(self.state.melds):
            # Normal one-card layoffs.
            for card in hand:
                combined = tuple(table_meld.cards) + (card,)
                if is_valid_layoff_result(combined):
                    actions.append(LayoffAction(meld_id=meld_id, cards=(card,)))

            # Special atomic bridge: singleton 7 -> three-card set of 7s.
            if len(table_meld.cards) == 1 and int(table_meld.cards[0].rank) == 7:
                for chosen in combinations(hand, 2):
                    if all(int(card.rank) == 7 for card in chosen):
                        combined = tuple(table_meld.cards) + tuple(chosen)
                        if is_valid_layoff_result(combined):
                            actions.append(
                                LayoffAction(
                                    meld_id=meld_id,
                                    cards=self._canonical_cards(chosen),
                                )
                            )
        return self._deduplicate_actions(actions)

    def legal_actions(self) -> list[Action]:
        s = self.state
        if s.terminal:
            return []

        player = s.current_player
        hand = s.hands[player]

        if s.phase == TurnPhase.DRAW:
            actions: list[Action] = []
            if self._can_draw_stock():
                actions.append(DrawStockAction())

            # Hoola rule used by this project: if the cards already in hand can
            # form a NEW registration, the player may register it immediately
            # instead of drawing.  We expose the actual MeldAction here rather
            # than a generic "skip draw" action, so skipping the draw cannot
            # be abused without making the required registration.
            if self.rules.allow_meld_before_draw:
                actions.extend(MeldAction(cards=m) for m in find_valid_melds(hand))

            # If the player has already registered, a legal layoff may also be
            # made immediately at turn start instead of drawing.
            if self.rules.allow_layoff_before_draw:
                actions.extend(self._layoff_actions(player, hand))

            if (
                self.rules.allow_thank_you
                and s.discard_pile
                and s.last_discarded_by is not None
                and s.last_discarded_by != player
            ):
                discarded = s.discard_pile[-1]
                for meld in valid_thank_you_melds(hand, discarded):
                    actions.append(ThankYouAction(meld_cards=meld))
            return self._deduplicate_actions(actions)

        if s.phase == TurnPhase.PLAY:
            actions = [MeldAction(cards=m) for m in find_valid_melds(hand)]

            actions.extend(self._layoff_actions(player, hand))

            # A player with cards remaining may stop playing melds and discard.
            actions.append(PassPlayAction())
            return self._deduplicate_actions(actions)

        if s.phase == TurnPhase.DISCARD:
            return [DiscardAction(card=c) for c in sorted(hand, key=lambda c: c.id)]

        return []

    @staticmethod
    def _deduplicate_actions(actions: list[Action]) -> list[Action]:
        # Frozen action dataclasses are hashable.
        return list(dict.fromkeys(actions))

    def step(self, action: Action) -> GameState:
        legal = self.legal_actions()
        if action not in legal:
            raise IllegalAction(f"Illegal action in {self.state.phase.name}: {action}")

        s = self.state
        player = s.current_player

        if isinstance(action, DrawStockAction):
            self._recycle_stock_if_needed()
            if not s.stock:
                self._end_draw()
                return s
            s.hands[player].append(s.stock.pop())
            s.phase = TurnPhase.PLAY
            return s

        if isinstance(action, ThankYouAction):
            discarded = s.discard_pile.pop()
            hand_cards = [c for c in action.meld_cards if c != discarded]
            # All legality was already checked by legal_actions().
            for card in hand_cards:
                s.hands[player].remove(card)
            s.melds.append(TableMeld(owner=player, cards=list(action.meld_cards)))
            s.has_registered[player] = True
            if not s.hands[player]:
                self._end_win(player)
            else:
                s.phase = TurnPhase.PLAY
            return s

        if isinstance(action, MeldAction):
            for card in action.cards:
                s.hands[player].remove(card)
            s.melds.append(TableMeld(owner=player, cards=list(action.cards)))
            s.has_registered[player] = True
            if not s.hands[player]:
                self._end_win(player)
            else:
                s.phase = TurnPhase.PLAY
            return s

        if isinstance(action, LayoffAction):
            for card in action.cards:
                s.hands[player].remove(card)
            table_meld = s.melds[action.meld_id]
            table_meld.cards.extend(action.cards)
            table_meld.cards.sort(key=lambda c: c.id)
            if not s.hands[player]:
                self._end_win(player)
            else:
                s.phase = TurnPhase.PLAY
            return s

        if isinstance(action, PassPlayAction):
            s.phase = TurnPhase.DISCARD
            return s

        if isinstance(action, DiscardAction):
            s.hands[player].remove(action.card)
            s.discard_pile.append(action.card)
            s.last_discarded_by = player
            s.turns_completed += 1

            if not s.hands[player]:
                self._end_win(player)
                return s

            if s.turns_completed >= self.max_turns:
                self._end_draw()
                return s

            s.current_player = 1 - player
            s.phase = TurnPhase.DRAW
            return s

        raise TypeError(f"Unsupported action type: {type(action)!r}")

    def _end_win(self, player: int) -> None:
        self.state.winner = player
        self.state.is_draw = False
        self.state.phase = TurnPhase.TERMINAL

    def _end_draw(self) -> None:
        self.state.winner = None
        self.state.is_draw = True
        self.state.phase = TurnPhase.TERMINAL

    def result(self) -> GameResult:
        if not self.state.terminal:
            raise RuntimeError("Game is not finished")
        return GameResult(
            winner=self.state.winner,
            is_draw=self.state.is_draw,
            turns_completed=self.state.turns_completed,
        )
