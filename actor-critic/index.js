module.exports = class ActorCritic {
  constructor({ v, policy, gamma = 1 }) {
    this.v = v
    this.policy = policy
    this.gamma = gamma
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = environment.getState()
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
    this.v.update(this.state, tdError)
    this.policy.update(this.state, this.action, tdError)
  }

  getTdError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.getGamma() * this.v.call(this.nextState)
    return this.environment.getReward() + nextEstimate - this.v.call(this.state)
  }

  getGamma() {
    return this.environment.gamma * this.gamma
  }
}
