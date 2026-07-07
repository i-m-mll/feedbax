export const semanticTokens = {
  selected: {
    border: 'border-brand-500',
    ring: 'ring-brand-500/40',
    strongRing: 'ring-brand-500/30',
    fill: 'bg-brand-600',
    softRing: 'ring-brand-300',
    text: 'text-brand-600',
  },
  error: {
    border: 'border-red-300',
    background: 'bg-red-50',
    text: 'text-red-600',
    badgeBackground: 'bg-red-100',
    badgeText: 'text-red-700',
    ring: 'ring-red-200',
  },
  warning: {
    border: 'border-amber-300',
    background: 'bg-amber-50',
    text: 'text-amber-700',
    badgeBackground: 'bg-amber-100',
    badgeText: 'text-amber-700',
    ring: 'ring-amber-200',
  },
  success: {
    background: 'bg-mint-100',
    text: 'text-mint-600',
  },
  objective: {
    fill: 'bg-violet-500',
    ring: 'ring-violet-200',
  },
  task: {
    fill: 'bg-emerald-500',
    ring: 'ring-emerald-200',
    line: 'bg-emerald-400',
    border: 'border-emerald-300',
    text: 'text-emerald-700',
  },
  highlighted: {
    border: 'border-amber-400',
    fill: 'bg-amber-400',
    ring: 'ring-amber-200',
  },
} as const;

