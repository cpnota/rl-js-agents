const math = require('mathjs')

module.exports = class MiniBatchStochasticGradientDescent {
  constructor({ miniBatchSize, epochs }) {
    this.miniBatchSize = miniBatchSize
    this.epochs = epochs
  }

  optimize({ samples, computeGradient, update }) {
    let count = 0
    let currentSampleIndex = 0
    while (count < samples.length * this.epochs) {
      let gradient = 0
      for (let i = 0; i < this.miniBatchSize; i++) {
        const sample = samples[currentSampleIndex]
        gradient = math.add(gradient, computeGradient(sample))
        count += 1
        currentSampleIndex += 1
        if (!samples[currentSampleIndex]) currentSampleIndex = 0
      }
      if (gradient === 0) {
        // we've either completely converged,
        // or there was some error.
        return
      }
      const direction = math.multiply(gradient, 1 / math.norm(gradient))
      update(direction)
    }
  }
}
