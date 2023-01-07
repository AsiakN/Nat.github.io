/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [".src/**/*.{html, js}", "*.{html, js}"],
  theme: {
    extend: {
      borderRadius: {
        xl: '1rem',
        '2xl': '2rem',
        '3xl': '3rem'
      }
    },
  },
  plugins: [],
  variants: {
    margin: ['responsive', 'hover']
  },
}
