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
    this.updateQ()
    this.state = this.nextState
    this.action = this.nextAction
  }

  updateQ() {
    const estimate = this.q.call(this.state, this.action)
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.score(this.nextState, this.nextAction)
    const error =
      this.environment.getReward() +
      this.environment.gamma * nextEstimate -
      estimate
    this.q.update(this.state, this.action, error)
  }
}
