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
    this.traces.update({
      state: this.state,
      action: this.action,
      tdError: this.getTDError(),
      decayAmount: this.lambda * this.environment.gamma
    })
  }

  getTDError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(this.nextState, this.nextAction)

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() +
      this.environment.gamma * nextEstimate -
      estimate
    )
  }
}
