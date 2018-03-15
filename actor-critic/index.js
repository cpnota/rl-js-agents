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
    const tdError =
      this.environment.getReward() +
      this.environment.gamma * this.v.call(this.nextState) -
      this.v.call(this.state)
    this.v.update(this.state, this.I * tdError)
    this.policy.update(this.state, this.action, this.I * tdError)
  }
}
