const math = require('mathjs')

/* eslint-disable camelcase */
module.exports = class NAC_TD {
  constructor({ policy, v, alpha_w }) {
    this.v = v
    this.policy = policy
    this.alpha_w = alpha_w
  }

  newEpisode(environment) {
    this.environment = environment
    this.vTraces = this.v.createTraces()
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
    const gradient = this.policy.partialLN(this.state, this.action)

    this.e_w = math.add(
      math.multiply(this.gamma, this.gamma, this.e_w),
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
      decayAmount: this.lambda * this.environment.gamma
    })
  }

  updateActor() {
    const norm = math.norm(this.w)
    this.policy.updateWeights(math.multiply(this.w, 1 / norm))
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
