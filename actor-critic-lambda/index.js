module.exports = class ActorCritic {
  constructor({ v, policy, lambda, gamma = 1 }) {
    this.v = v
    this.policy = policy
    this.lambda = lambda
    this.gamma = gamma
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = environment.getState()
    this.vTraces = this.v.createTraces()
    this.policyTraces = this.policy.createTraces()
  }

  act() {
    this.action = this.policy.chooseAction(this.state)
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
    this.update()
    this.state = this.nextState
  }

  update() {
    const tdError = this.getTdError()

    this.vTraces.record(this.state)
    this.vTraces.updateV(tdError)
    this.vTraces.decay(this.lambda * this.getGamma())

    this.policyTraces.record(this.state)
    this.policyTraces.updatePolicy(tdError)
    this.policyTraces.decay(this.lambda * this.getGamma())
  }

  getTdError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.environment.gamma * this.v.call(this.nextState)
    return this.environment.getReward() + nextEstimate - this.v.call(this.state)
  }

  getGamma() {
    return this.environment.gamma * this.gamma
  }
}
