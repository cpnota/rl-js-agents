module.exports = class ActorCritic {
  constructor({ v, policy, lambda }) {
    this.v = v
    this.policy = policy
    this.lambda = lambda
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = environment.getState()
    this.I = 1
    this.vTraces = this.v.createTraces()
    this.policyTraces = this.policy.createTraces()
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
    const tdError =
      this.environment.getReward() +
      this.environment.gamma * this.v.call(this.nextState) -
      this.v.call(this.state)
    this.vTraces.update({
      state: this.state,
      tdError,
      decayAmount: this.lambda * this.environment.gamma
    })
    this.policyTraces.update({
      state: this.state,
      action: this.action,
      tdError,
      decayAmount: this.lambda * this.environment.gamma
    })
  }
}
