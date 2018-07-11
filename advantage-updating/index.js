const Agent = require('@rl-js/interfaces/agent')

module.exports = class Sarsa extends Agent {
  constructor({ a, v, policy, omega, alpha }) {
    super()
    this.a = a
    this.v = v
    this.policy = policy
    this.omega = omega
    this.alpha = alpha
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
    const aMaxOld = this.getAMax(this.state)
    const aActual = this.a.call(this.state, this.action)
    const tdError = this.getTdError()
    this.updateA(aMaxOld, aActual, tdError)
    const aMaxNew = this.getAMax(this.state)
    this.updateV(aMaxNew, aMaxOld)
    this.normalize()
  }

  updateA(aMaxOld, aAction, tdError) {
    const error = aMaxOld - aAction + tdError
    this.a.update(this.state, this.action, error)
  }

  updateV(aMaxNew, aMaxOld) {
    const error = aMaxNew - aMaxOld
    this.v.update(this.state, error)
  }

  getAMax(state) {
    const bestAction = this.policy.chooseBestAction(state)
    return this.a.call(this.state, bestAction)
  }

  normalize() {
    // TODO random state/action?
    const aMax = this.getAMax(this.state)
    this.a.update(this.state, this.action, (-this.omega / this.alpha) * aMax)
  }

  getTdError() {
    const nextEstimate = this.environment.isTerminated()
      ? 0
      : this.v.call(this.nextState)

    const estimate = this.v.call(this.state)

    return (
      this.environment.getReward() +
      this.environment.gamma * nextEstimate -
      estimate
    )
  }
}
