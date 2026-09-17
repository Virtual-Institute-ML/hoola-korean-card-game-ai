import type { CardData } from '../types'

type Props = {
  card?: CardData
  hidden?: boolean
  selected?: boolean
  clickable?: boolean
  onClick?: () => void
  compact?: boolean
}

export function Card({ card, hidden, selected, clickable, onClick, compact }: Props) {
  const red = card?.suit === 1 || card?.suit === 2
  const classes = [
    'card',
    hidden ? 'card-back' : '',
    red ? 'red' : '',
    selected ? 'selected' : '',
    clickable ? 'clickable' : '',
    compact ? 'compact' : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button className={classes} disabled={!clickable} onClick={onClick} aria-label={card?.text || 'hidden card'}>
      {hidden ? <span className="back-mark">H</span> : <span>{card?.text}</span>}
    </button>
  )
}
