module.exports = class Sarsa {
  constructor({ q, policy, lambda }) {
    this.q = q
    this.policy = policy
    this.lambda = lambda
  }

  newEpisode(environment) {
    this.environment = environment
    this.traces = this.q.createTraces()
    this.state = this.environment.getState()
    this.action = this.policy.chooseAction(this.state)
    this.qOld = 0
  }

  act() {
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
    this.nextAction = this.policy.chooseAction(this.nextState)
    this.update()
    this.state = this.nextState
    this.action = this.nextAction
  }

  update() {
    const q = this.q.call(this.state, this.action)
    const qNext = this.environment.isTerminated()
      ? 0
      : this.q.call(this.nextState, this.nextAction)

    const tdError =
      this.environment.getReward() + this.environment.gamma * qNext - q
    const offset = q - this.qOld
    const decayAmount = this.lambda * this.environment.gamma

    this.traces.trueOnlineUpdate({
      state: this.state,
      action: this.action,
      tdError,
      offset,
      decayAmount
    })

    this.qOld = qNext
  }
}
