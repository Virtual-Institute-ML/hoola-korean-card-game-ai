export type CardData = {
  id: number
  text: string
  rank: number
  suit: number
}

export type LegalAction = {
  id: number
  type: string
  label: string
  card_ids?: number[]
  meld_id?: number
}

export type MeldData = {
  id: number
  owner: number
  cards: CardData[]
}

export type GameState = {
  game: {
    terminal: boolean
    winner: number | null
    is_draw: boolean
    turns_completed: number
    phase: string
    current_player: number
    human_player: number
    human_turn: boolean
    opponent: string
    seed: number
  }
  human: {
    hand: CardData[]
    registered: boolean
  }
  opponent: {
    hand_size: number
    registered: boolean
  }
  table: {
    stock_size: number
    discard: CardData[]
    melds: MeldData[]
  }
  legal_actions: LegalAction[]
  action_log: { player: number; actor: string; text: string }[]
  recording: {
    enabled: boolean
    saved: boolean
    npz_file: string | null
    json_file: string | null
  }
}

export type GameResponse = {
  game_id: string
  state: GameState
}
