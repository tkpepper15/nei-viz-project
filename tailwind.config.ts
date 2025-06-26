import type { Config } from "tailwindcss";

export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        primary: "var(--primary)",
        "primary-dark": "var(--primary-hover)",
        "primary-hover": "var(--primary-hover)",
        danger: "#ef4444",
        "danger-light": "rgba(239, 68, 68, 0.1)",
        neutral: {
          50: "var(--neutral-50)",
          100: "var(--neutral-100)",
          200: "var(--neutral-200)",
          700: "var(--neutral-700)",
          800: "var(--neutral-800)",
          900: "var(--neutral-900)",
        }
      },
      backdropFilter: {
        'none': 'none',
        'blur': 'blur(20px)',
      },
      animation: {
        'spin': 'spin 1s linear infinite',
      }
    },
  },
  plugins: [],
  // Safelist to preserve our custom CSS classes
  safelist: [
    // Circuit components
    'circuit-container',
    'circuit-header', 
    'circuit-content',
    'circuit-sidebar',
    'circuit-tabs',
    'circuit-tab',
    'circuit-tab-active',
    'circuit-tab-inactive',
    'circuit-main',
    'circuit-visualization',
    'circuit-status',
    // Custom buttons
    'button-primary',
    'button-secondary',
    'button-success',
    // Spider plot
    'spider-visualization',
    'spider-plot',
    'spider-svg',
    'spider-card',
    'spider-card-header',
    'spider-value-indicator',
    'spider-heading',
    'spider-label',
    'spider-parameter-label',
    'spider-parameter-name',
    'spider-parameter-value',
    'spider-parameter-unit',
    'spider-reference-dot',
    'spider-grid-line',
    'spider-axis-line',
    'spider-control-label',
    'spider-control-value',
    'spider-range-slider',
    'spider-info-box',
    'spider-info-title',
    'spider-info-row',
    'spider-info-label',
    // Slider components
    'param-slider',
    'slider-label',
    'slider-value',
    'slider-input',
    'slider-label-text',
    'slider-tick-label',
    'slider-tick-level',
    // Data table
    'data-table',
    'data-table-header',
    'data-th',
    'data-td',
    // Utility classes
    'icon-button',
    'card',
    'card-header',
    'collapsible-section',
    'section-header',
    'section-content',
    'text-title',
    'text-subtitle',
    'text-small',
    'text-data',
    // Pattern matching for responsive variants
    {
      pattern: /^(circuit|spider|button|param|slider|data)-.*$/,
    }
  ]
} satisfies Config;
