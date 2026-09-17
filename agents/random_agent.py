from __future__ import annotations

import random

from hoola.action import Action, MeldAction
from hoola.meld import is_single_seven
from hoola.observation import Observation


class RandomAgent:
    """
    Mostly-random baseline agent.

    A single 7 is a legal Hoola meld, but treating it exactly like every other
    PLAY action makes a purely random agent shed isolated 7s too aggressively.
    To keep the baseline visibly random without that artifact, single-7 melds
    receive a smaller sampling weight by default.

    This is an agent policy choice, not a game-rule change: single 7s remain
    fully legal in the engine.
    """

    def __init__(
        self,
        seed: int | None = None,
        single_seven_weight: float = 0.15,
    ) -> None:
        if single_seven_weight < 0:
            raise ValueError("single_seven_weight must be >= 0")
        self.rng = random.Random(seed)
        self.single_seven_weight = single_seven_weight

    def _weight(self, action: Action) -> float:
        if isinstance(action, MeldAction) and is_single_seven(action.cards):
            return self.single_seven_weight
        return 1.0

    def select_action(self, observation: Observation, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions available")

        weights = [self._weight(action) for action in legal_actions]
        if not any(weight > 0 for weight in weights):
            raise RuntimeError("All legal-action sampling weights are zero")

        return self.rng.choices(legal_actions, weights=weights, k=1)[0]
