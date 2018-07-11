const Agent = require('@rl-js/interfaces/agent')

module.exports = class QLearning extends Agent {
  constructor({ q, policy, lambda, gamma = 1 }) {
    super()
    this.q = q
    this.policy = policy
    this.lambda = lambda
    this.gamma = gamma
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
    this.traces.record(this.state, this.action)
    this.traces.updateQ(this.getTDError(bestAction))

    if (bestAction !== this.nextAction) {
      // Reset the eligibility trace.
      // Otherwise, we backpropagate information about suboptimal actions.
      // This is the difference between q-learning and sarsa,
      // q-learning seeks to learn the q-function for the OPTIMAL policy.
      this.traces = this.q.createTraces()
    } else {
      this.traces.decay(this.lambda * this.getGamma())
    }
  }

  getTDError(bestAction) {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.q.call(this.nextState, bestAction)

    const estimate = this.q.call(this.state, this.action)

    return (
      this.environment.getReward() + this.getGamma() * nextEstimate - estimate
    )
  }

  getGamma() {
    return this.environment.gamma * this.gamma
  }
}
