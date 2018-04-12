const { Matrix } = require('ml-matrix')

module.exports = {
  outer: (v1, v2) => v1.map(x1 => v2.map(x2 => x1 * x2)),
  pinv: A => new Matrix(A).pseudoInverse().to2DArray()
}
