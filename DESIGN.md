# AgroDash Design System: The Digital Agronomist

## 1. Creative North Star
**"The Digital Agronomist"** — A blend of precision data science and organic, tactile agriculture. 
- **Intentional Asymmetry**: Use large-scale typography and overlapping elements.
- **Tonal Layering**: Depth is achieved through background color shifts, not rigid grids or heavy shadows.

## 2. Color Palette (Material 3 Tonal Range)

| Token | Hex | Usage |
| :--- | :--- | :--- |
| **Primary** | `#00450d` | "Fertile Soil." Critical CTAs and brand moments. |
| **Primary Container** | `#1b5e20` | Interactive states and primary icons. |
| **Tertiary** | `#503600` | "Golden Harvest." Price alerts and predicted upward trends. |
| **Tertiary Fixed Dim** | `#ffba38` | Accent moments for data highlights. |
| **Surface** | `#f8faf8` | "Off-white" base to reduce eye strain. |
| **Surface Container Low** | `#f2f4f2` | Standard background for page sections. |
| **Surface Container Lowest**| `#ffffff` | Primary cards (provides a "lift"). |
| **On Surface** | `#191c1b` | Main text (never use pure black). |
| **Outline Variant** | `#c0c9bb` | Used at 20% opacity for "Ghost Borders." |

## 3. Typography
- **Headings**: `Inter Bold` (Tight letter-spacing, editorial feel).
- **Body**: `Inter Regular` (Analytical descriptions).
- **Data/Numbers**: `Roboto Mono` (Ticker tape precision for prices, percentages, timestamps).

## 4. Layout & UI Rules
- **The "No-Line" Rule**: DO NOT use 1px solid borders. Define boundaries via background color shifts.
- **Glassmorphism**: 80% opacity `surface-container-lowest` with `backdrop-filter: blur(12px)` for floating tooltips.
- **Elevation**: Achievement of depth by stacking (e.g., `surface-container-lowest` card on `surface-container-low` background).
- **Dividers**: Forbidden. Use vertical whitespace (spacing scale) or subtle hover states.

## 5. Component Specifics
- **Buttons**: `lg` (8px) radius, subtle inner-glow gradient for primary actions.
- **Input Fields**: `surface-container-low`, no border. On focus: 2px left vertical accent bar.
- **Charts**: 
    - Historical: Solid `primary` line.
    - Predicted: Dashed `tertiary` line.
    - Fill: Gradient from `primary` (20%) to transparent.
