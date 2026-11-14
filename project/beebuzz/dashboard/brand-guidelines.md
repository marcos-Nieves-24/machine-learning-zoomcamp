# BeeBuzz — Brand Guidelines (v1)

## Logo

### Primary mark
- **File:** `logo.svg` (symbol, 256×256)
- **Colors:** Amber `#f5a623` on dark `#0f1117`
- **Concept:** Hexagon (honeycomb) + 3 sound waves (buzz) + center dot (bee)
- **Audit score:** 99/100

### Lockups
- **Horizontal:** `logo-horizontal.svg` (symbol + wordmark, 400×128)
- **Stacked:** symbol above wordmark
- **Symbol only:** `logo.svg` (for favicon, app icon, small spaces)
- **Wordmark only:** "BeeBuzz" in Inter Bold

### Variants (in `variants/`)
- `logo-black.svg` — black on white (print)
- `logo-white.svg` — white on dark (reversed)
- `logo-mono-f5a623.svg` — monochrome amber
- `logo-square.svg` — square crop
- `logo-favicon.svg` — simplified for 16px
- `logo-app-icon.svg` — app icon (1024px)

### Color palette

| Name | HEX | RGB | Usage |
|---|---|---|---|
| Amber | `#f5a623` | 245, 166, 35 | Primary accent, logo, CTAs |
| Dark | `#0f1117` | 15, 17, 23 | Background, logo bg |
| Dark surface | `#1a1a2e` | 26, 26, 46 | Cards, panels |
| Border | `#2a2a40` | 42, 42, 64 | Borders, dividers |
| Text primary | `#e0e0e0` | 224, 224, 224 | Headings |
| Text secondary | `#9ca3af` | 156, 163, 175 | Body, captions |
| Text muted | `#555555` | 85, 85, 85 | Footer, hints |

### Typography
- **Primary:** Inter (Bold for headings, Regular for body)
- **Fallback:** 'Helvetica Neue', Arial, sans-serif

### Clear space
- Minimum clear space around the logo = 1/4 of the logo height on all sides.
- For the symbol (256px): 64px clear space.

### Minimum sizes
- **Symbol only:** 16px (favicon) — use `logo-small.svg` or `logo-favicon.svg`
- **Horizontal lockup:** 120px wide minimum
- **Full logo with wordmark:** 200px wide minimum

### Approved backgrounds
- Dark (#0f1117) — preferred
- White (#ffffff) — use `logo-black.svg`
- Amber (#f5a623) — use `logo-white.svg`

### Misuse (do NOT)
- Do not stretch or distort the logo
- Do not change the colors (except approved variants)
- Do not add shadows, gradients, or effects
- Do not rotate the logo
- Do not place on busy backgrounds without clear space
- Do not use the wordmark without the symbol (except in text contexts)

### Notes
- The wordmark uses `<text>` elements. For production, convert to paths (outlines).
- Strokes are used for the sound waves. For print/embroidery, expand to filled outlines.
- The logo works in one colour (amber on dark, or black on white).
