from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import HeuristicAgent
from hoola.action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from hoola.action_codec import ActionCodec
from hoola.encoding import encode_observation
from hoola.engine import HoolaEngine
from hoola.observation import Observation, make_observation
from hoola.training_recorder import TrainingRecorder
from rl.model import load_policy_checkpoint
from rl.ppo import resolve_device


class NewGameRequest(BaseModel):
    opponent: str = "heuristic"  # heuristic | rl
    human_player: int = 0
    seed: int | None = None
    rl_checkpoint: str | None = None
    rl_device: str = "cpu"


class ActionRequest(BaseModel):
    action_id: int


class RLCheckpointAgent:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_policy_checkpoint(
            checkpoint_path, device=self.device
        )
        self.codec = ActionCodec()

    def select_action(self, observation: Observation, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions available")
        obs = torch.as_tensor(
            encode_observation(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask = torch.as_tensor(
            self.codec.action_mask(legal_actions), dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        action_id, _, _ = self.model.act(obs, mask, deterministic=True)
        return self.codec.decode_legal(int(action_id.item()), legal_actions)


@dataclass
class GameSession:
    engine: HoolaEngine
    human_player: int
    ai_name: str
    ai_agent: Any
    seed: int
    action_log: list[dict[str, Any]]
    training_recorder: TrainingRecorder | None = None
    training_record_path: Path | None = None
    training_record_saved: bool = False
    training_meta_path: Path | None = None


app = FastAPI(title="Hoola AI Web API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, GameSession] = {}
CODEC = ActionCodec()
DEFAULT_RL_CHECKPOINT = os.getenv("HOOLA_RL_CHECKPOINT")
WEB_RECORD_DIR = Path(os.getenv("HOOLA_WEB_RECORD_DIR", "records/web"))
WEB_RECORDING_ENABLED = os.getenv("HOOLA_WEB_RECORDING", "1").strip().lower() not in {
    "0", "false", "no", "off"
}


def card_json(card) -> dict[str, Any]:
    return {
        "id": card.id,
        "text": str(card),
        "rank": int(card.rank),
        "suit": int(card.suit),
    }


def action_label(action: Action, observation: Observation) -> str:
    if isinstance(action, DrawStockAction):
        return "카드 뽑기"
    if isinstance(action, ThankYouAction):
        taken = str(observation.discard_pile[-1]) if observation.discard_pile else "?"
        cards = " ".join(str(c) for c in action.meld_cards)
        return f"땡큐: {taken} 가져와서 {cards} 등록"
    if isinstance(action, MeldAction):
        cards = " ".join(str(c) for c in action.cards)
        suffix = " (드로우 없이)" if observation.phase.name == "DRAW" else ""
        return f"등록: {cards}{suffix}"
    if isinstance(action, LayoffAction):
        cards = " ".join(str(c) for c in action.cards)
        owner = "내" if observation.melds[action.meld_id].owner == observation.player_id else "상대"
        suffix = " (드로우 없이)" if observation.phase.name == "DRAW" else ""
        return f"붙이기: {cards} → #{action.meld_id} {owner} 멜드{suffix}"
    if isinstance(action, PassPlayAction):
        return "플레이 종료 → 버리기"
    if isinstance(action, DiscardAction):
        return f"{action.card} 버리기"
    return str(action)


def action_json(action: Action, observation: Observation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": CODEC.encode(action),
        "type": action.type.name,
        "label": action_label(action, observation),
    }
    if isinstance(action, DiscardAction):
        payload["card_ids"] = [action.card.id]
    elif isinstance(action, MeldAction):
        payload["card_ids"] = [c.id for c in action.cards]
    elif isinstance(action, ThankYouAction):
        payload["card_ids"] = [c.id for c in action.meld_cards]
    elif isinstance(action, LayoffAction):
        payload["card_ids"] = [c.id for c in action.cards]
        payload["meld_id"] = action.meld_id
    return payload



def make_training_record_path(*, opponent: str, game_id: str) -> Path:
    """Create a collision-resistant filename for one completed web game.

    The HumanDataset/heuristic loaders only care about the NPZ + JSON contents,
    so timestamp + short UUID names are preferable to global counters for a web
    app where multiple games may exist at once.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    short_id = game_id.split("-")[0]
    return WEB_RECORD_DIR / f"web_human_vs_{opponent}_{stamp}_{short_id}.npz"


def save_training_record_if_finished(session: GameSession) -> None:
    """Save exactly once, and only after the game reaches a terminal state."""
    if session.training_recorder is None:
        return
    if session.training_record_saved:
        return
    if not session.engine.state.terminal:
        return

    npz_path, meta_path = session.training_recorder.save(session.engine.state)
    session.training_record_path = npz_path
    session.training_meta_path = meta_path
    session.training_record_saved = True


def step_and_record(
    session: GameSession,
    *,
    player: int,
    acting_agent: str,
    observation: Observation,
    legal_actions: list[Action],
    action: Action,
) -> None:
    """Apply one action and mirror the CLI TrainingRecorder contract exactly."""
    append_log(session, player, action)
    session.engine.step(action)
    if session.training_recorder is not None:
        session.training_recorder.record_decision(
            acting_player=player,
            acting_agent=acting_agent,
            observation=observation,
            legal_actions=legal_actions,
            selected_action=action,
            state_after=session.engine.state,
        )
    save_training_record_if_finished(session)

def state_json(session: GameSession) -> dict[str, Any]:
    engine = session.engine
    human = session.human_player
    obs = make_observation(engine.state, human)

    human_turn = not engine.state.terminal and engine.state.current_player == human
    legal_actions = engine.legal_actions() if human_turn else []

    return {
        "game": {
            "terminal": engine.state.terminal,
            "winner": engine.state.winner,
            "is_draw": engine.state.is_draw,
            "turns_completed": engine.state.turns_completed,
            "phase": engine.state.phase.name,
            "current_player": engine.state.current_player,
            "human_player": human,
            "human_turn": human_turn,
            "opponent": session.ai_name,
            "seed": session.seed,
        },
        "human": {
            "hand": [card_json(c) for c in obs.my_hand],
            "registered": obs.has_registered[human],
        },
        "opponent": {
            "hand_size": obs.opponent_hand_size,
            "registered": obs.has_registered[1 - human],
        },
        "table": {
            "stock_size": obs.stock_size,
            "discard": [card_json(c) for c in obs.discard_pile],
            "melds": [
                {
                    "id": idx,
                    "owner": meld.owner,
                    "cards": [card_json(c) for c in meld.cards],
                }
                for idx, meld in enumerate(obs.melds)
            ],
        },
        "legal_actions": [action_json(a, obs) for a in legal_actions],
        "action_log": session.action_log[-12:],
        "recording": {
            "enabled": session.training_recorder is not None,
            "saved": session.training_record_saved,
            "npz_file": (
                session.training_record_path.name
                if session.training_record_saved and session.training_record_path is not None
                else None
            ),
            "json_file": (
                session.training_meta_path.name
                if session.training_record_saved and session.training_meta_path is not None
                else None
            ),
        },
    }


def describe_action(action: Action) -> str:
    if isinstance(action, DrawStockAction):
        return "stock에서 1장 뽑음"
    if isinstance(action, ThankYouAction):
        return "땡큐: " + " ".join(str(c) for c in action.meld_cards)
    if isinstance(action, MeldAction):
        return "등록: " + " ".join(str(c) for c in action.cards)
    if isinstance(action, LayoffAction):
        return f"붙이기 #{action.meld_id}: " + " ".join(str(c) for c in action.cards)
    if isinstance(action, PassPlayAction):
        return "플레이 종료"
    if isinstance(action, DiscardAction):
        return f"{action.card} 버림"
    return str(action)


def append_log(session: GameSession, player: int, action: Action) -> None:
    session.action_log.append(
        {
            "player": player,
            "actor": "human" if player == session.human_player else session.ai_name,
            "text": describe_action(action),
        }
    )


def advance_ai(session: GameSession) -> None:
    engine = session.engine
    ai_player = 1 - session.human_player

    safety = 0
    while not engine.state.terminal and engine.state.current_player == ai_player:
        safety += 1
        if safety > 100:
            raise RuntimeError("AI turn exceeded safety limit")
        legal = engine.legal_actions()
        if not legal:
            engine._end_draw()
            save_training_record_if_finished(session)
            break
        obs = make_observation(engine.state, ai_player)
        action = session.ai_agent.select_action(obs, legal)
        step_and_record(
            session,
            player=ai_player,
            acting_agent=session.ai_name,
            observation=obs,
            legal_actions=legal,
            action=action,
        )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "sessions": len(SESSIONS)}


@app.post("/api/games")
def new_game(request: NewGameRequest) -> dict[str, Any]:
    if request.human_player not in (0, 1):
        raise HTTPException(400, "human_player must be 0 or 1")
    if request.opponent not in {"heuristic", "rl"}:
        raise HTTPException(400, "opponent must be 'heuristic' or 'rl'")

    seed = request.seed if request.seed is not None else secrets.randbits(63)
    engine = HoolaEngine(seed=seed)

    if request.opponent == "heuristic":
        ai_agent = HeuristicAgent(seed=seed + 200)
    else:
        checkpoint = request.rl_checkpoint or DEFAULT_RL_CHECKPOINT
        if not checkpoint:
            raise HTTPException(
                400,
                "RL checkpoint not configured. Set HOOLA_RL_CHECKPOINT or provide rl_checkpoint.",
            )
        if not Path(checkpoint).exists():
            raise HTTPException(400, f"RL checkpoint not found: {checkpoint}")
        ai_agent = RLCheckpointAgent(checkpoint, device=request.rl_device)

    game_id = str(uuid.uuid4())

    training_recorder = None
    training_record_path = None
    if WEB_RECORDING_ENABLED:
        training_record_path = make_training_record_path(
            opponent=request.opponent, game_id=game_id
        )
        player_types = [request.opponent, request.opponent]
        player_types[request.human_player] = "human"
        player_types[1 - request.human_player] = request.opponent
        metadata: dict[str, Any] = {
            "source": "web",
            "game_id": game_id,
            "seed": seed,
            "human_player": request.human_player,
            "opponent": request.opponent,
            "player_types": player_types,
        }
        if request.opponent == "rl":
            metadata["rl_checkpoint"] = str(
                request.rl_checkpoint or DEFAULT_RL_CHECKPOINT
            )
            metadata["rl_update"] = getattr(ai_agent, "checkpoint", {}).get(
                "update", None
            )
        training_recorder = TrainingRecorder(
            training_record_path, metadata=metadata
        )

    session = GameSession(
        engine=engine,
        human_player=request.human_player,
        ai_name=request.opponent,
        ai_agent=ai_agent,
        seed=seed,
        action_log=[],
        training_recorder=training_recorder,
        training_record_path=training_record_path,
    )
    SESSIONS[game_id] = session

    # Supports human playing P1 as well.
    advance_ai(session)
    return {"game_id": game_id, "state": state_json(session)}


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict[str, Any]:
    session = SESSIONS.get(game_id)
    if session is None:
        raise HTTPException(404, "Game not found")
    return {"game_id": game_id, "state": state_json(session)}


@app.post("/api/games/{game_id}/actions")
def play_action(game_id: str, request: ActionRequest) -> dict[str, Any]:
    session = SESSIONS.get(game_id)
    if session is None:
        raise HTTPException(404, "Game not found")

    engine = session.engine
    if engine.state.terminal:
        raise HTTPException(409, "Game is already finished")
    if engine.state.current_player != session.human_player:
        raise HTTPException(409, "It is not the human player's turn")

    legal = engine.legal_actions()
    try:
        action = CODEC.decode_legal(request.action_id, legal)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    obs = make_observation(engine.state, session.human_player)
    step_and_record(
        session,
        player=session.human_player,
        acting_agent="human",
        observation=obs,
        legal_actions=legal,
        action=action,
    )
    advance_ai(session)
    return {"game_id": game_id, "state": state_json(session)}


@app.delete("/api/games/{game_id}")
def delete_game(game_id: str) -> dict[str, Any]:
    if game_id not in SESSIONS:
        raise HTTPException(404, "Game not found")
    del SESSIONS[game_id]
    return {"ok": True}
