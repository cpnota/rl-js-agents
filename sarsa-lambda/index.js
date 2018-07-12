const Agent = require('@rl-js/interfaces/agent')

module.exports = class Sarsa extends Agent {
  constructor({ q, policy, lambda, traces, gamma = 1 }) {
    super()
    this.q = q
    this.policy = policy
    this.lambda = lambda
    this.gamma = gamma
    this.traces = traces
  }

  newEpisode(environment) {
    this.traces.reset()
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
    this.traces
      .record(this.state, this.action)
      .update(this.getTDError())
      .decay(this.lambda * this.getGamma())
  }

  getTDError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(this.nextState, this.nextAction)

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() + this.getGamma() * nextEstimate - estimate
    )
  }

  getGamma() {
    return this.environment.gamma * this.gamma
  }
}
