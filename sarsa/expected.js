const math = require('mathjs')

module.exports = class ExpectedSarsa {
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
    this.q.update(this.state, this.action, this.getTDError())
    this.state = this.nextState
    this.action = this.nextAction
  }

  getTDError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.getExpected()

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() +
      this.environment.gamma * nextEstimate -
      estimate
    )
  }

  getExpected() {
    return math.sum(
      ...this.policy.actions.map(action => {
        const value = this.q.call(this.nextState, action)
        const probability = this.policy.getProbability(this.nextState, action)
        return value * probability
      })
    )
  }
}
