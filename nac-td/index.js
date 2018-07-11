const math = require('mathjs')
const CompatibleAdvantageEstimator = require('./advantage')
const Agent = require('@rl-js/interfaces/agent')

/* eslint-disable camelcase */
module.exports = class NAC_TD extends Agent {
  constructor({ policy, v, alpha_w, update_frequency, lambda, gamma = 1 }) {
    super()
    this.v = v
    this.policy = policy
    this.alpha_w = alpha_w
    this.update_frequency = update_frequency
    this.lambda = lambda
    this.gamma = gamma

    this.count = 0
    this.compatibleAdvantageEstimator = new CompatibleAdvantageEstimator({
      policy,
      alpha: alpha_w
    })
  }

  newEpisode(environment) {
    this.environment = environment
    this.vTraces = this.v.createTraces()
    this.advantageTraces = this.compatibleAdvantageEstimator.createTraces()
  }

  act() {
    this.execute()
    this.updateCritic()
    this.updateActor()
  }

  execute() {
    this.state = this.environment.getState()
    this.action = this.policy.chooseAction(this.state)
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
  }

  updateCritic() {
    const tdError = this.getTDError()
    const estimatedAdvantage = this.compatibleAdvantageEstimator.call(
      this.state,
      this.action
    )
    const advantageError = tdError - estimatedAdvantage
    const decayFactor = this.getGamma() * this.lambda

    this.advantageTraces.record(this.state, this.action)
    this.advantageTraces.updateAdvantage(advantageError)
    this.advantageTraces.decay(decayFactor)

    this.vTraces.record(this.state)
    this.vTraces.updateV(tdError)
    this.vTraces.decay(decayFactor)
  }

  updateActor() {
    this.count += 1
    if (this.count === this.update_frequency) {
      const weights = this.compatibleAdvantageEstimator.weights
      const norm = math.norm(weights)
      const normalizedWeights = math.multiply(this.weights, 1 / norm)
      this.policy.updateWeights(normalizedWeights)
      this.count = 0
    }
  }

  getTDError() {
    const reward = this.environment.getReward()
    const nextValue = this.environment.isTerminated()
      ? 0
      : this.v.call(this.nextState)
    const previousValue = this.v.call(this.state)
    return reward + this.getGamma() * nextValue - previousValue
  }

  getGamma() {
    return this.environment.gamma * this.gamma
  }
}
