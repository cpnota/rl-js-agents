module.exports = class ActorCritic {
  constructor({ v, policy }) {
    this.v = v
    this.policy = policy
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = environment.getState()
    this.I = 1
  }

  act() {
    this.action = this.policy.chooseAction(this.state)
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
    this.update()
    // this.I *= this.environment.gamma
    this.state = this.nextState
  }

  update() {
    const tdError = this.getTdError()
    this.v.update(this.state, this.I * tdError)
    this.policy.update(this.state, this.action, this.I * tdError)
  }

  getTdError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.environment.gamma * this.v.call(this.nextState)
    return this.environment.getReward() + nextEstimate - this.v.call(this.state)
  }
}
