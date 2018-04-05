const math = require('mathjs')
const { outer, pinv } = require('./lin')

/* TODO eligibility traces */
/* eslint-disable camelcase */
module.exports = class QLearning {
  constructor({ q, policy, basis, gamma = 1, alpha_gain, lambda }) {
    this.q = q
    this.policy = policy
    this.basis = basis
    this.gamma = gamma
    this.alpha_gain = alpha_gain
    this.lambda = lambda
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = this.environment.getState()
    this.features = this.basis.features(this.state)
    this.traces = this.features
    this.A = math.zeros(this.basis.terms, this.basis.terms)._data
  }

  act() {
    this.action = this.policy.chooseAction(this.state)
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
    this.nextFeatures = this.basis.features(this.nextState)
    this.update()
    this.state = this.nextState
    this.features = this.nextFeatures
  }

  update() {
    const tdError = this.getTDError()
    const diff = math.subtract(
      math.multiply(this.getBeta(), this.nextFeatures),
      this.features
    )
    const A = outer(this.traces, diff)
    this.A = math.add(
      this.A,
      math.multiply(this.alpha_gain, math.subtract(A, this.A))
    )
    const errors = math.multiply(pinv(this.A), this.traces, -tdError)
    this.q.getApproximator(this.action).updateWeights(errors)
    this.traces = math.multiply(this.lambda, this.getBeta(), this.traces)
    this.traces = math.add(this.traces, this.nextFeatures)
  }

  getTDError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(
          this.nextState,
          this.policy.chooseBestAction(this.nextState)
        )

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() + this.getBeta() * nextEstimate - estimate
    )
  }

  getBeta() {
    return this.gamma * this.environment.gamma
  }
}
