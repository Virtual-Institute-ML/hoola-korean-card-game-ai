from __future__ import annotations

import math
import random
from functools import lru_cache
from itertools import combinations

from hoola.action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from hoola.card import Card, Rank, Suit, create_deck
from hoola.meld import find_valid_melds, is_single_seven
from hoola.observation import Observation


def format_cards_for_explain(cards) -> str:
    return " ".join(str(card) for card in sorted(cards, key=lambda c: c.id))


class HeuristicAgent:
    """
    Explainable one-ply Hoola baseline.

    v1 principles
    -------------
    1. Reduce hand size aggressively with melds / layoffs / Thank You.
    2. When several plays overlap, preserve the remainder with the strongest
       set/run potential.
    3. When discarding, keep cards that form useful rank pairs or near-runs.
    4. Penalize discards that are more likely to give the opponent Thank You,
       using *public information only*.

    The policy never inspects GameState, the stock contents, or the opponent's
    hidden hand.  It receives only Observation + legal_actions, the same API
    intended for future learned agents.
    """

    # Immediate card reduction dominates, while hand potential breaks close
    # strategic choices between overlapping melds.
    REMOVE_CARD_BONUS = 24.0
    REGISTER_BONUS = 5.0
    THANK_YOU_BONUS = 12.0

    # Discard risk matters more when the opponent is close to going out.
    DISCARD_RISK_BASE = 16.0
    DISCARD_RISK_ENDGAME = 7.0

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, observation: Observation, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions available")

        scored = [(self.score_action(observation, action), action) for action in legal_actions]
        best_score = max(score for score, _ in scored)
        best = [action for score, action in scored if math.isclose(score, best_score, abs_tol=1e-9)]
        return self.rng.choice(best)

    def ranked_actions(self, observation: Observation, legal_actions: list[Action]):
        """Return legal actions sorted from highest to lowest heuristic score."""
        scored = [(self.score_action(observation, action), action) for action in legal_actions]
        return sorted(scored, key=lambda item: item[0], reverse=True)

    def explain_decision(
        self,
        observation: Observation,
        legal_actions: list[Action],
        selected_action: Action,
        *,
        mode: str = "public",
        top_k: int = 5,
    ) -> str:
        """Explain one heuristic decision.

        mode='public' is deliberately non-revealing and suitable for a fair
        human-vs-AI game. mode='full' lists candidate actions and scores and is
        intended for development/debugging because it can reveal hidden-hand
        information.
        """
        selected_score = self.score_action(observation, selected_action)
        if mode == "public":
            return self._public_reason(selected_action, selected_score)
        if mode != "full":
            raise ValueError("mode must be 'public' or 'full'")

        ranked = self.ranked_actions(observation, legal_actions)
        lines = [
            "HEURISTIC v1 FULL DEBUG",
            f"  selected: {self._action_text(selected_action)} | score={selected_score:.3f}",
            "  top candidates:",
        ]
        for rank, (score, action) in enumerate(ranked[:top_k], start=1):
            flag = " <- chosen" if action == selected_action else ""
            lines.append(f"    {rank}. {self._action_text(action):<42} score={score:9.3f}{flag}")
        return "\n".join(lines)

    @staticmethod
    def _action_text(action: Action) -> str:
        if isinstance(action, DrawStockAction):
            return "DRAW from stock"
        if isinstance(action, ThankYouAction):
            return f"THANK YOU -> {format_cards_for_explain(action.meld_cards)}"
        if isinstance(action, MeldAction):
            return f"MELD {format_cards_for_explain(action.cards)}"
        if isinstance(action, LayoffAction):
            return f"LAYOFF {format_cards_for_explain(action.cards)} -> #{action.meld_id}"
        if isinstance(action, PassPlayAction):
            return "PASS"
        if isinstance(action, DiscardAction):
            return f"DISCARD {action.card}"
        return repr(action)

    @staticmethod
    def _public_reason(action: Action, score: float) -> str:
        prefix = f"Heuristic v1 chose {HeuristicAgent._action_text(action)}."
        if isinstance(action, DrawStockAction):
            reason = "No legal Thank You was preferred, so it takes the normal stock draw."
        elif isinstance(action, ThankYouAction):
            reason = "Thank You immediately converts the public discard into a legal meld and reduces hand size."
        elif isinstance(action, MeldAction):
            reason = "The meld reduces hand size while preserving the heuristic's remaining-hand value; at DRAW phase it also avoids an unnecessary draw."
        elif isinstance(action, LayoffAction):
            reason = "Laying off safely removes cards from the hand using an existing public meld."
        elif isinstance(action, PassPlayAction):
            reason = "Keeping the current hand structure scored better than the available immediate plays."
        elif isinstance(action, DiscardAction):
            reason = "This discard gave the best balance of remaining-hand structure and estimated Thank You risk."
        else:
            reason = "It had the highest heuristic preference among legal actions."
        return f"{prefix} {reason} [score={score:.3f}]"

    def score_action(self, observation: Observation, action: Action) -> float:
        """Return an explainable scalar preference score for one legal action."""
        hand = tuple(observation.my_hand)

        if isinstance(action, DrawStockAction):
            # The identity of the stock card is hidden, so a one-ply heuristic
            # cannot value it directly.  Zero is the neutral DRAW baseline.
            return 0.0

        if isinstance(action, ThankYouAction):
            remaining = self._remaining_after_thank_you(hand, action, observation)
            removed = len(hand) - len(remaining)
            if not remaining:
                return 1_000_000.0
            return (
                self.THANK_YOU_BONUS
                + self.REMOVE_CARD_BONUS * removed
                + self._hand_quality(remaining)
                + self.REGISTER_BONUS
            )

        if isinstance(action, MeldAction):
            remaining = self._remove_cards(hand, action.cards)
            removed = len(action.cards)
            if not remaining:
                return 1_000_000.0

            score = self.REMOVE_CARD_BONUS * removed + self._hand_quality(remaining)
            if not observation.has_registered[observation.player_id]:
                score += self.REGISTER_BONUS

            # A singleton 7 is genuinely useful because it safely removes one
            # card and opens layoff access.  We do not artificially suppress it
            # here (the RandomAgent suppression is only a baseline-policy fix).
            # Still, overlapping 3+ card melds naturally beat it via removed-card
            # count and remainder quality.
            if is_single_seven(action.cards):
                score += 1.0
            return score

        if isinstance(action, LayoffAction):
            remaining = self._remove_cards(hand, action.cards)
            removed = len(action.cards)
            if not remaining:
                return 1_000_000.0
            return self.REMOVE_CARD_BONUS * removed + self._hand_quality(remaining) + 2.0

        if isinstance(action, PassPlayAction):
            # Passing keeps all cards, so it should lose to a clearly useful
            # meld/layoff but remains legal when preserving the hand is better.
            return self._hand_quality(hand)

        if isinstance(action, DiscardAction):
            remaining = self._remove_cards(hand, (action.card,))
            if not remaining:
                return 1_000_000.0

            risk = self.estimate_thank_you_risk(observation, action.card)
            endgame_pressure = max(0, 4 - observation.opponent_hand_size)
            risk_weight = self.DISCARD_RISK_BASE + self.DISCARD_RISK_ENDGAME * endgame_pressure

            return self._hand_quality(remaining) - risk_weight * risk

        raise TypeError(f"Unsupported action type: {type(action)!r}")

    # ------------------------------------------------------------------
    # Hand evaluation
    # ------------------------------------------------------------------
    @classmethod
    def _hand_quality(cls, hand: tuple[Card, ...]) -> float:
        """
        Value cards that work together.

        Components:
        - exact disjoint meld coverage already present in the hand;
        - same-rank pairs that can become a set;
        - same-suit adjacent / one-gap pairs that can become a run.
        """
        if not hand:
            return 0.0

        quality = 7.0 * cls._max_meld_coverage(hand)

        for a, b in combinations(hand, 2):
            if a.rank == b.rank:
                quality += 5.0

            if a.suit == b.suit:
                distance = cls._cyclic_rank_distance(a.rank, b.rank)
                if distance == 1:
                    quality += 4.0
                elif distance == 2:
                    quality += 1.75

        return quality

    @staticmethod
    def _cyclic_rank_distance(a: Rank, b: Rank) -> int:
        da = abs(int(a) - int(b))
        return min(da, 13 - da)

    @classmethod
    def _max_meld_coverage(cls, hand: tuple[Card, ...]) -> int:
        """Maximum number of cards coverable by disjoint legal melds."""
        cards = tuple(sorted(hand, key=lambda c: c.id))
        if not cards:
            return 0

        index = {card: i for i, card in enumerate(cards)}
        masks: list[int] = []
        for meld in find_valid_melds(cards):
            mask = 0
            for card in meld:
                mask |= 1 << index[card]
            masks.append(mask)

        @lru_cache(maxsize=None)
        def best(used: int) -> int:
            answer = 0
            for mask in masks:
                if used & mask:
                    continue
                answer = max(answer, mask.bit_count() + best(used | mask))
            return answer

        return best(0)

    # ------------------------------------------------------------------
    # Public-information opponent-risk model
    # ------------------------------------------------------------------
    @classmethod
    def estimate_thank_you_risk(cls, observation: Observation, candidate: Card) -> float:
        """
        Approximate P(opponent can Thank You | candidate is discarded).

        No hidden information is used.  We remove all publicly known cards and
        our own hand from the deck, then consider the six possible *two-card*
        partner patterns that can make an immediate 3-card meld with candidate:

        - 3 choices of two matching-rank cards (set)
        - 3 choices of two same-suit ranks (3-card run containing candidate)

        If U cards are unknown and the opponent holds n of them, the probability
        that one particular required pair is in the opponent hand is

            n(n-1) / [U(U-1)].

        Pair events overlap, so summing them is an approximation; the result is
        clipped to [0, 1].  This is intentionally lightweight for v1.
        """
        n = observation.opponent_hand_size
        if n < 2:
            return 0.0

        known: set[Card] = set(observation.my_hand)
        known.update(observation.discard_pile)
        for meld in observation.melds:
            known.update(meld.cards)

        unknown = set(create_deck()) - known
        u = len(unknown)
        if u < 2 or n > u:
            return 0.0

        partner_pairs = cls._thank_you_partner_pairs(candidate)
        available_pairs = sum(1 for a, b in partner_pairs if a in unknown and b in unknown)
        if available_pairs == 0:
            return 0.0

        p_specific_pair = (n * (n - 1)) / (u * (u - 1))
        return min(1.0, available_pairs * p_specific_pair)

    @staticmethod
    def _thank_you_partner_pairs(candidate: Card) -> set[tuple[Card, Card]]:
        pairs: set[tuple[Card, Card]] = set()

        # Same-rank set: choose any two of the other three suits.
        same_rank = [
            Card(suit, candidate.rank)
            for suit in Suit
            if suit != candidate.suit
        ]
        for a, b in combinations(same_rank, 2):
            pairs.add(tuple(sorted((a, b), key=lambda c: c.id)))

        # Same-suit 3-card cyclic runs containing candidate.  With rank r,
        # the possible partner offsets are (-2,-1), (-1,+1), (+1,+2).
        r = int(candidate.rank)

        def card_at(offset: int) -> Card:
            wrapped_rank = Rank(((r - 1 + offset) % 13) + 1)
            return Card(candidate.suit, wrapped_rank)

        for offsets in ((-2, -1), (-1, 1), (1, 2)):
            a, b = card_at(offsets[0]), card_at(offsets[1])
            pairs.add(tuple(sorted((a, b), key=lambda c: c.id)))

        return pairs

    # ------------------------------------------------------------------
    # Small state-transition helpers operating only on Observation
    # ------------------------------------------------------------------
    @staticmethod
    def _remove_cards(hand: tuple[Card, ...], cards) -> tuple[Card, ...]:
        remaining = list(hand)
        for card in cards:
            remaining.remove(card)
        return tuple(sorted(remaining, key=lambda c: c.id))

    @classmethod
    def _remaining_after_thank_you(
        cls,
        hand: tuple[Card, ...],
        action: ThankYouAction,
        observation: Observation,
    ) -> tuple[Card, ...]:
        discarded = observation.discard_pile[-1] if observation.discard_pile else None
        hand_cards = [card for card in action.meld_cards if card != discarded]
        return cls._remove_cards(hand, hand_cards)
