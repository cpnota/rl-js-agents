const { outer, pinv } = require('./lin')
const math = require('mathjs')

test('multiplies 2x2', () => {
  expect(outer([1, 2], [3, 4])).toEqual([[3, 4], [6, 8]])
})

test('takes pseudoinverse', () => {
  const res = pinv([[1, 2], [3, 4]])
  expect(res[0][0]).toBeCloseTo(-2, 3)
  expect(res[0][1]).toBeCloseTo(1, 3)
  expect(res[1][0]).toBeCloseTo(1.5, 3)
  expect(res[1][1]).toBeCloseTo(-0.5, 3)
})
