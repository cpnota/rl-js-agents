const Traces = require('./traces')

// constructs advantage estimator compatible with the given policy
module.exports = class CompatibleAdvantageEstimator {
  constructor ({ policy, alpha }) {
    this.alpha = alpha
    this.policy = policy
    this.weights = 0 // lazy initialize
  }

  call(state, action) {
    const features = this.policy.partialLN(state, action)
    if (!this.weights) return 0
    return math.dot(this.weights, features)
  }

  updateWeights(weights, error) {
    this.weights = math.add(
      this.weights,
      math.multiply(this.alpha, error, weights)
    )
  }

  createTraces() {
    return new Traces(this)
  }
}
