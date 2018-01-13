module.exports = {
  env: {
    es6: true,
    node: true
  },
  extends: [
    'eslint:recommended',
    'plugin:import/recommended',
    'plugin:mocha/recommended',
    'plugin:node/recommended',
    'plugin:promise/recommended',
    'prettier-standard'
  ],
  parserOptions: {
    ecmaVersion: 2017
  },
  plugins: [
    'import',
    'mocha',
    'node',
    'prettier',
    'promise', // needed for standard
    'standard'
  ],
  rules: {
    'import/unambiguous': ['off'],
    'quote-props': ['warn', 'consistent-as-needed'],
    'sort-keys': ['warn']
  }
}
