const vp = require('./vector-product')
const math = require('mathjs')

test('multiplies 2x2', () => {
  expect(vp([1, 2], [3, 4])).toEqual([
    [3, 4],
    [6, 8]
  ])
})
