const math = require('mathjs')

/* eslint-disable camelcase */
module.exports = class NAC_TD {
  constructor({ policy, v, alpha_w, update_frequency }) {
    this.v = v
    this.policy = policy
    this.alpha_w = alpha_w
    this.count = 0
    this.update_frequency = update_frequency
  }

  newEpisode(environment) {
    this.environment = environment
    this.vTraces = this.v.createTraces()
    this.e_w = undefined // TODO remove this hack
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

    if (this.e_w === undefined) {
      // TODO this hack is horrific
      this.e_w = this.policy.partialLN(this.state, this.action).fill(0)
    }
  }

  updateCritic() {
    const tdError = this.getTDError()
    const gradient = this.policy.partialLN(this.state, this.action)

    this.e_w = math.add(
      math.multiply(this.getGamma(), this.lambda, this.e_w),
      gradient
    )

    const tdEstimate = math.dot(this.w, gradient)
    const tdErrorError = tdError - tdEstimate

    this.w = math.add(
      this.w,
      math.multiply(this.alpha_w, tdErrorError, this.e_w)
    )

    this.vTraces.update({
      state: this.state,
      tdError,
      decayAmount: this.lambda * this.getGamma()
    })
  }

  updateActor() {
    this.count = this.count + 1
    if (this.count === this.update_frequency) {
      const norm = math.norm(this.w)
      this.policy.updateWeights(math.multiply(this.w, 1 / norm))
      this.count = 0
    }
  }

  getGamma() {
    return this.environment.gamma
  }

  getTDError() {
    const reward = this.environment.getReward()
    const nextValue = this.environment.isTerminated()
      ? 0
      : this.v.score(this.nextState)
    const previousValue = this.v.score(this.state)
    return reward + this.getGamma() * nextValue - previousValue
  }
}
