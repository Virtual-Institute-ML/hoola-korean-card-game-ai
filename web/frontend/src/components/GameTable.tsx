import { useMemo } from 'react'
import type { GameState, LegalAction } from '../types'
import { Card } from './Card'

type Props = {
  state: GameState
  busy: boolean
  onAction: (id: number) => void
}

const groupTitle: Record<string, string> = {
  DRAW_STOCK: '드로우',
  THANK_YOU: '땡큐',
  MELD: '등록',
  LAYOFF: '붙이기',
  PASS_PLAY: '플레이 종료',
  DISCARD: '버리기',
}

export function GameTable({ state, busy, onAction }: Props) {
  const actionsByType = useMemo(() => {
    const groups = new Map<string, LegalAction[]>()
    for (const action of state.legal_actions) {
      const list = groups.get(action.type) || []
      list.push(action)
      groups.set(action.type, list)
    }
    return groups
  }, [state.legal_actions])

  const drawAction = state.legal_actions.find((a) => a.type === 'DRAW_STOCK')
  const discardByCard = new Map<number, LegalAction>()
  for (const action of state.legal_actions) {
    if (action.type === 'DISCARD' && action.card_ids?.length === 1) {
      discardByCard.set(action.card_ids[0], action)
    }
  }

  const topDiscard = state.table.discard.at(-1)
  const human = state.game.human_player

  return (
    <div className="game-shell">
      <header className="topbar">
        <div>
          <div className="brand">HOOLA AI</div>
          <div className="subtitle">vs {state.game.opponent === 'rl' ? 'RL v1' : 'Heuristic v1'}</div>
        </div>
        <div className="turn-chip">
          {state.game.terminal
            ? '게임 종료'
            : state.game.human_turn
              ? `내 차례 · ${state.game.phase}`
              : 'AI 생각 중…'}
        </div>
      </header>

      <section className="opponent-zone">
        <div className="zone-label">
          AI · {state.opponent.hand_size}장 {state.opponent.registered ? '· 등록 완료' : ''}
        </div>
        <div className="opponent-hand">
          {Array.from({ length: state.opponent.hand_size }).map((_, i) => (
            <Card key={i} hidden compact />
          ))}
        </div>
      </section>

      <section className="table-zone">
        <div className="pile-row">
          <button
            className={`pile stock ${drawAction && !busy ? 'clickable-pile' : ''}`}
            disabled={!drawAction || busy}
            onClick={() => drawAction && onAction(drawAction.id)}
          >
            <div className="mini-card-back">H</div>
            <span>STOCK</span>
            <strong>{state.table.stock_size}</strong>
          </button>

          <div className="pile discard-pile">
            {topDiscard ? <Card card={topDiscard} /> : <div className="empty-card">—</div>}
            <span>DISCARD</span>
          </div>
        </div>

        <div className="meld-area">
          {state.table.melds.length === 0 ? (
            <div className="empty-melds">등록된 카드가 없습니다.</div>
          ) : (
            state.table.melds.map((meld) => (
              <div className="meld" key={meld.id}>
                <div className="meld-label">
                  #{meld.id} · {meld.owner === human ? '나' : 'AI'}
                </div>
                <div className="meld-cards">
                  {meld.cards.map((card) => (
                    <Card key={card.id} card={card} compact />
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="actions-zone">
        {state.game.terminal ? (
          <div className="result-wrap">
            <div className="result-banner">
              {state.game.is_draw
                ? '무승부'
                : state.game.winner === human
                  ? '🎉 승리했습니다!'
                  : 'AI가 승리했습니다.'}
            </div>
            {state.recording.enabled && state.recording.saved && (
              <div className="record-saved">
                학습 데이터 저장 완료 · {state.recording.npz_file}
              </div>
            )}
          </div>
        ) : (
          <>
            {[...actionsByType.entries()]
              .filter(([type]) => type !== 'DISCARD' && type !== 'DRAW_STOCK')
              .map(([type, actions]) => (
                <div className="action-group" key={type}>
                  <div className="action-title">{groupTitle[type] || type}</div>
                  <div className="action-buttons">
                    {actions.map((action) => (
                      <button disabled={busy} key={action.id} onClick={() => onAction(action.id)}>
                        {action.label}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
          </>
        )}
      </section>

      <section className="human-zone">
        <div className="zone-label">
          내 패 · {state.human.hand.length}장 {state.human.registered ? '· 등록 완료' : ''}
        </div>
        <div className="human-hand">
          {state.human.hand.map((card) => {
            const discard = discardByCard.get(card.id)
            return (
              <Card
                key={card.id}
                card={card}
                clickable={Boolean(discard) && !busy}
                onClick={() => discard && onAction(discard.id)}
              />
            )
          })}
        </div>
        {discardByCard.size > 0 && <div className="hint">버릴 카드는 직접 클릭하세요.</div>}
      </section>

      <details className="history">
        <summary>최근 행동</summary>
        {state.action_log.length === 0 ? (
          <p>아직 행동 기록이 없습니다.</p>
        ) : (
          <ol>
            {state.action_log.map((item, i) => (
              <li key={i}>
                <b>{item.actor === 'human' ? '나' : 'AI'}:</b> {item.text}
              </li>
            ))}
          </ol>
        )}
      </details>
    </div>
  )
}
