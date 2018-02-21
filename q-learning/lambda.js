module.exports = class QLearning {
  constructor({ q, policy, lambda }) {
    this.q = q
    this.policy = policy
    this.lambda = lambda
  }

  newEpisode(environment) {
    this.environment = environment
    this.state = this.environment.getState()
    this.action = this.policy.chooseAction(this.state)
    this.traces = this.q.createTraces()
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
    const bestAction = this.policy.chooseBestAction(this.nextState)
    const tdError = this.getTDError(bestAction)
    this.traces.updateEligibility(this.state, this.action)
    this.traces.updateQ(tdError)
    if (bestAction === this.nextAction) {
      this.traces.decay(this.lambda * this.environment.gamma)
    } else {
      this.traces = this.act.createTraces() // reset
    }
  }

  getTDError(bestAction) {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(this.nextState, bestAction)

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() +
      this.environment.gamma * nextEstimate -
      estimate
    )
  }
}
