const { Agent, ActionValueFunction, ActionTraces, Policy, } = require('@rl-js/interfaces')
const checkInterface = require('check-interface')
const check = require('check-types')

module.exports = class Sarsa extends Agent {
  constructor({ actionValueFunction, policy, actionTraces, lambda, gamma = 1 }) {
    super()
    this.actionValueFunction = checkInterface(actionValueFunction, ActionValueFunction)
    this.actionTraces = checkInterface(actionTraces, ActionTraces)
    this.policy = checkInterface(policy, Policy)
    this.lambda = check.assert.number(lambda)
    this.gamma = check.assert.number(gamma)
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
      : this.actionValueFunction.call(this.nextState, this.nextAction)

    const estimate = this.actionValueFunction.call(this.state, this.action)

    return (
      this.environment.getReward() + this.getGamma() * nextEstimate - estimate
    )
  }

  getGamma() {
    return this.environment.gamma * this.gamma
  }
}
