const math = require('mathjs')

module.exports = class Sarsa {
  constructor({ q, policy }) {
    this.q = q
    this.policy = policy
  }

  newEpisode(environment) {
    this.environment = environment
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
    const tdError = this.getTDError()
    const errors = math.multiply(
      tdError,
      math.subtract(
        this.q.gradient(this.state, this.action),
        math.multiply(
          this.environment.gamma,
          this.q.gradient(this.nextState, this.nextAction)
        )
      )
    )
    this.q.updateWeights(errors)
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
