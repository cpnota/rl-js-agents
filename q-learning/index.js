const Agent = require('@rl-js/interfaces/agent')

module.exports = class QLearning extends Agent {
  constructor({ q, policy }) {
    super()
    this.q = q
    this.policy = policy
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = this.environment.getState()
  }

  act() {
    this.action = this.policy.chooseAction(this.state)
    this.environment.dispatch(this.action)
    this.nextState = this.environment.getState()
    this.q.update(this.state, this.action, this.getTDError())
    this.state = this.nextState
  }

  getTDError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(
          this.nextState,
          this.policy.chooseBestAction(this.nextState)
        )

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() +
      this.environment.gamma * nextEstimate -
      estimate
    )
  }
}
