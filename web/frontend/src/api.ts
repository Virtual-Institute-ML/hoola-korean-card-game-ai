import type { GameResponse } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail || `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function createGame(opponent: 'heuristic' | 'rl'): Promise<GameResponse> {
  return request('/api/games', {
    method: 'POST',
    body: JSON.stringify({ opponent, human_player: 0 }),
  })
}

export function playAction(gameId: string, actionId: number): Promise<GameResponse> {
  return request(`/api/games/${gameId}/actions`, {
    method: 'POST',
    body: JSON.stringify({ action_id: actionId }),
  })
}
