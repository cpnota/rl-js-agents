const math = require('mathjs')
const { outer, pinv } = require('./lin')

/* TODO eligibility traces */
/* eslint-disable camelcase */
module.exports = class QLearning {
  constructor({ q, policy, basis, gamma = 1, gain, lambda }) {
    this.q = q
    this.policy = policy
    this.basis = basis
    this.gamma = gamma
    this.gain = gain
    this.lambda = lambda
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = this.environment.getState()
    this.traces = undefined
    this.A = undefined
  }

  act() {
    this.action = this.policy.chooseAction(this.state)
    this.features = this.q.features(this.state, this.action)
    this.traces = this.traces
      ? math.add(this.traces, this.features)
      : this.features
    this.A = this.A || math.zeros(this.traces.length, this.traces.length)._data
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
    this.update()
    this.state = this.nextState
  }

  update() {
    const bestAction = this.policy.chooseBestAction(this.nextState)
    const tdError = this.getTDError(bestAction)
    const nextFeatures = this.q.features(this.nextState, bestAction)

    const diff = math.subtract(
      math.multiply(this.getBeta(), nextFeatures),
      this.features
    )
    const A = outer(this.traces, diff)
    this.A = math.add(
      this.A,
      math.multiply(this.gain, math.subtract(A, this.A))
    )
    // const bias = math.multiply(0.001, math.eye(A.length))._data // TODO remove magic number
    const A_inv = pinv(A)
    const errors = math.multiply(A_inv, this.traces, -tdError)
    this.q.updateWeights(errors)
    this.traces = math.multiply(this.lambda, this.getBeta(), this.traces)
  }

  getTDError(bestAction) {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(this.nextState, bestAction)

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() + this.getBeta() * nextEstimate - estimate
    )
  }

  getBeta() {
    return this.gamma * this.environment.gamma
  }
}
