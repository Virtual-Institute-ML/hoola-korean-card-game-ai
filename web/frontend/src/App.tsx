import { useState } from 'react'
import { createGame, playAction } from './api'
import { GameTable } from './components/GameTable'
import type { GameState } from './types'
import './styles.css'

function App() {
  const [gameId, setGameId] = useState<string | null>(null)
  const [state, setState] = useState<GameState | null>(null)
  const [opponent, setOpponent] = useState<'heuristic' | 'rl'>('heuristic')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function newGame() {
    setBusy(true)
    setError(null)
    try {
      const response = await createGame(opponent)
      setGameId(response.game_id)
      setState(response.state)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function act(actionId: number) {
    if (!gameId) return
    setBusy(true)
    setError(null)
    try {
      const response = await playAction(gameId, actionId)
      setState(response.state)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  if (!state) {
    return (
      <main className="landing">
        <div className="landing-card">
          <div className="logo-mark">H</div>
          <h1>HOOLA AI</h1>
          <p>인공지능과 2인 훌라를 플레이해보세요.</p>
          <div className="opponent-picker">
            <button className={opponent === 'heuristic' ? 'active' : ''} onClick={() => setOpponent('heuristic')}>
              Heuristic v1
            </button>
            <button className={opponent === 'rl' ? 'active' : ''} onClick={() => setOpponent('rl')}>
              RL v1
            </button>
          </div>
          <button className="primary" disabled={busy} onClick={newGame}>
            {busy ? '게임 준비 중…' : '새 게임 시작'}
          </button>
          {error && <div className="error">{error}</div>}
        </div>
      </main>
    )
  }

  return (
    <main className="app-page">
      <div className="toolbar">
        <button onClick={() => setState(null)}>← 메뉴</button>
        <button disabled={busy} onClick={newGame}>새 게임</button>
      </div>
      {error && <div className="error floating-error">{error}</div>}
      <GameTable state={state} busy={busy} onAction={act} />
    </main>
  )
}

export default App
