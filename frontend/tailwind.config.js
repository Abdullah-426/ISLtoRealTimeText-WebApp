module.exports = {
    darkMode: ['class', 'class'],
    content: [
        './app/**/*.{ts,tsx}',
        './components/**/*.{ts,tsx}'
    ],
    theme: {
        extend: {
            colors: {
                background: '#0b0b12',
                foreground: '#e5e7eb',
                primary: '#7C3AED',
                secondary: '#06B6D4'
            },
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui'],
                display: ['Poppins', 'ui-sans-serif', 'system-ui']
            },
            borderRadius: {
                '2xl': '1rem'
            }
        }
    },
    plugins: []
}
