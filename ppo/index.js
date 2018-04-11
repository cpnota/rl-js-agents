const math = require('mathjs')

module.exports = class TrustRegionPolicyOptimization {
  constructor({ policy, v }) {
    this.policy = policy
    this.v = v
  }

  newEpisode(environment) {
    this.environment = environment
    this.history = []
  }

  act() {
    const state = this.environment.getState()
    const action = this.policy.chooseAction(state)
    const actionProbabilities = this.policy.getProbabilities(state)
    this.environment.dispatch(action)
    const reward = this.environment.getReward()
    this.history.push({ state, action, reward, actionProbabilities })

    if (this.environment.isTerminated()) {
      this.update()
    }
  }

  update() {
    const returns = this.getReturns()
    const baselines = this.getBaselines()
    const advantages = returns.map((r, i) => r - baselines[i])
    this.updateBaselines(advantages)
    const policyGradients = this.policyGradient(advantages)
  }

  getReturns() {
    const rewards = this.history.map(({ reward }) => reward)
    return rewards.map((reward, index) =>
      this.getDiscountedReturn(rewards.slice(index))
    )
  }

  getDiscountedReturn(rewards) {
    const discountedRewards = rewards.map(
      (reward, index) => Math.pow(this.getGamma(), index) * reward
    )
    return math.add(0, ...discountedRewards) // mathjs gets mad if you only pass one value
  }

  getBaselines() {
    return this.history.map(({ state }) => this.v.call(state))
  }

  updateBaselines(errors) {
    this.history.forEach(({ state }, i) => this.v.update(state, errors[i]))
  }

  policyGradient(errors) {
    const gradients = this.history.map(({ state, action }, i) =>
      math.multiply(this.policy.partialLN(state, action), errors[i])
    )
    return math.add(...gradients)
  }

  getGamma() {
    return 1
  }
}
