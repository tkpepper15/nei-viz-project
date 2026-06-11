/**
 * SpideyPlot canvas UI design tokens.
 *
 * Single source of truth for every floating surface, button, node, and
 * typographic element on the canvas. Import from here — never hard-code
 * color or spacing values directly in component files.
 *
 * Color naming convention:
 *   bg-*    — background fills, from darkest to lightest
 *   border-* — border/divider fills
 *   text-*   — foreground text fills
 *   accent-* — category accent colors (node port dots, edge strokes)
 *
 * Class-string constants (PILL, ICON_BTN, …) are complete Tailwind strings
 * so the Tailwind scanner can detect every class statically.
 */

// ---------------------------------------------------------------------------
// Color palette (use in inline `style` props or JS logic)
// ---------------------------------------------------------------------------

export const COLORS = {
  // Backgrounds — layered from deepest to most elevated
  bgCanvas:    '#0d0d0f',   // raw canvas
  bgRaised:    '#131316',   // pills, HUDs, node bodies
  bgElement:   '#1e1e24',   // active button fill, inner node header
  bgHover:     '#232329',   // button pressed / deeper hover
  bgSubtle:    '#1a1a1f',   // ghost hover on inactive buttons

  // Borders
  borderFaint:  '#1e1e24',  // node inner section dividers
  borderDefault:'#2a2a33',  // pill outline, dividers between HUD sections
  borderStrong: '#3a3a46',  // active / focused button border
  borderFocus:  '#4a4a56',  // hover on active buttons

  // Text
  textPrimary:  '#dddde2',
  textSecondary:'#9a9aa2',
  textMuted:    '#7a7a82',
  textFaint:    '#454549',
  textGhost:    '#2a2a33',  // disabled / placeholder

  // Node-category accent colors (source → process → viz → other)
  accentSource:  '#4ade80',
  accentProcess: '#60a5fa',
  accentViz:     '#f97316',
  accentOther:   '#818cf8',

  // Semantic status
  statusOk:    '#3d7a60',   // toggle active green
} as const;

// ---------------------------------------------------------------------------
// Floating surface container — same for both bottom toolbar and top HUD
// ---------------------------------------------------------------------------

/** Pill container shell — add positioning + gap yourself. */
export const PILL =
  'bg-[#131316] border border-[#2a2a33] rounded-xl shadow-xl shadow-black/40';

/** Standard inner padding for the pill. */
export const PILL_PAD = 'px-2 py-1.5';

/** Full pill container: combine PILL + PILL_PAD + flex. */
export const PILL_BASE =
  'flex items-center gap-1 bg-[#131316] border border-[#2a2a33] rounded-xl px-2 py-1.5 shadow-xl shadow-black/40';

/** Thin vertical divider between HUD sections. */
export const DIVIDER =
  'w-px h-5 bg-[#2a2a33] mx-0.5 flex-shrink-0 self-center';

// ---------------------------------------------------------------------------
// Button variants — all sized to fit inside a pill
// ---------------------------------------------------------------------------

/** Icon-only square button (28 × 28 px). */
export const ICON_BTN =
  'w-7 h-7 flex items-center justify-center rounded-lg text-[#9a9aa2] hover:text-[#dddde2] hover:bg-[#1e1e24] transition-colors duration-100 focus:outline-none flex-shrink-0';

/** Icon-only button, active/toggled state. */
export const ICON_BTN_ACTIVE =
  'w-7 h-7 flex items-center justify-center rounded-lg border border-[#3a3a46] bg-[#1e1e24] text-[#dddde2] hover:border-[#4a4a56] hover:bg-[#232329] transition-colors duration-100 focus:outline-none flex-shrink-0';

/** Label button — inactive / ghost state. */
export const LABEL_BTN_INACTIVE =
  'flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border border-transparent text-[#454549] hover:text-[#7a7a82] hover:bg-[#1a1a1f] transition-colors duration-100 focus:outline-none';

/** Label button — active / primary state. */
export const LABEL_BTN_ACTIVE =
  'flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border border-[#3a3a46] bg-[#1e1e24] text-[#dddde2] hover:border-[#4a4a56] hover:bg-[#232329] transition-colors duration-100 focus:outline-none';

/** Label button — disabled / upcoming feature. */
export const LABEL_BTN_DISABLED =
  'flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg text-[#2a2a33] cursor-not-allowed';

/** Monospace display value (zoom %, counts). */
export const MONO_VALUE =
  'px-2 h-7 text-[10px] font-mono text-[#9a9aa2] hover:text-[#dddde2] hover:bg-[#1e1e24] rounded-lg transition-colors focus:outline-none tabular-nums min-w-[3rem] text-center flex-shrink-0';

// ---------------------------------------------------------------------------
// Node chrome — headers, bodies, borders
// ---------------------------------------------------------------------------

/** Outer node container. */
export const NODE_CONTAINER =
  'flex flex-col rounded-lg border border-[#1e1e24]';

/** Node header bar (default). */
export const NODE_HEADER =
  'flex-shrink-0 flex items-center gap-2 px-3 py-1.5 border-b border-[#1e1e24] rounded-t-lg cursor-grab select-none transition-colors duration-150 bg-[#131316]';

/** Node header bar (focused). */
export const NODE_HEADER_FOCUSED =
  'flex-shrink-0 flex items-center gap-2 px-3 py-1.5 border-b border-[#1e1e24] rounded-t-lg cursor-grab select-none transition-colors duration-150 bg-[#18181c]';

/** Node body padding. */
export const NODE_BODY = 'p-3 flex flex-col gap-2.5 text-[10px] text-[#7a7a82]';

/** Node section divider. */
export const NODE_DIVIDER = 'border-t border-[#1e1e24]';

// ---------------------------------------------------------------------------
// Icon sizes — use on the svg element's className
// ---------------------------------------------------------------------------

export const ICON = {
  /** Node-header chrome (trash, chevron indicators) — 10px */
  node: 'w-2.5 h-2.5',
  /** Compact pill / banner icons — 12px */
  sm:   'w-3 h-3',
  /** Pill toolbar button icons — 14px */
  md:   'w-3.5 h-3.5',
  /** Panel / section feature icons — 16px */
  lg:   'w-4 h-4',
  /** Large modal / alert icons — 20px */
  xl:   'w-5 h-5',
} as const;

// ---------------------------------------------------------------------------
// Node chrome measurements (px) — kept in sync with NODE_HEADER padding
// Update these if header padding or control-bar heights change.
// ---------------------------------------------------------------------------

export const NODE_CHROME = {
  /** Outer header bar height: py-1.5 × 2 (12px) + toggle height (14px) + border (1px) */
  headerH:   28,
  /** Progress bar strip (h-px) */
  progressH:  1,
  /** Inner control bar used by VizTrajectoryBody */
  ctrlBarH:  38,
  /** Extra padding budget absorbed by chart wrappers / scroll gutters */
  chartPadH: 40,
} as const;

// ---------------------------------------------------------------------------
// Typography scale
// ---------------------------------------------------------------------------

export const TYPE = {
  /** Brand / app name */
  brand:        'text-sm font-semibold text-[#dddde2] tracking-tight',
  /** Node header label */
  nodeLabel:    'text-[10px] font-medium text-[#dddde2]',
  /** Button label */
  btnLabel:     'text-xs',
  /** Monospace numeric value */
  mono:         'text-[10px] font-mono tabular-nums',
  /** Node progress counter (running indicator) */
  nodeProgress: 'text-[10px] font-mono tabular-nums text-[#454549] flex-shrink-0',
  /** Section / group heading */
  sectionHd:    'text-[9px] uppercase tracking-widest text-[#454549]',
  /** Body / description copy */
  body:         'text-xs text-[#7a7a82] leading-relaxed',
  /** Faint helper text */
  helper:       'text-[10px] text-[#454549]',
} as const;

// ---------------------------------------------------------------------------
// Danger / delete button — for node header trash action
// ---------------------------------------------------------------------------

export const NODE_DELETE_BTN =
  'text-[#2a2a33] hover:text-[#c45a5a] transition-colors duration-100 focus:outline-none flex-shrink-0';
