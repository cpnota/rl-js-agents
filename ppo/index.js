const math = require('mathjs')

// FIXME right now this is just REINFORCE with baselines
module.exports = class ProximalPolicyOptimization {
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
    const actionProbability = this.policy.getProbability(state, action)
    this.environment.dispatch(action)
    const reward = this.environment.getReward()
    this.history.push({ state, action, reward, actionProbability })

    if (this.shouldUpdate()) {
      this.update()
    }
  }

  shouldUpdate() {
    // TODO real criteria
    return this.environment.isTerminated()
  }

  update() {
    this.updateValueFunction()
    this.computeAdvantages()
    this.optimizePolicy()
  }

  updateValueFunction() {
    this.history.forEach(({ state, reward }, t) => {
      const estimate = this.v.call(state)

      const nextEstimate = this.history[t + 1]
        ? this.v.call(this.history[t + 1].state)
        : 0

      const tdError = reward + this.getGamma() * nextEstimate - estimate

      this.v.update(state, tdError)
    })
  }

  computeAdvantages() {
    this.computeReturns()
    this.computeBaselines()

    for (const step of this.history) {
      step.advantage = step.return - step.baseline
    }
  }

  optimizePolicy() {
    let gradient = 0

    for (const step of this.history) {
      gradient = math.add(gradient, this.getSampleGradient(step))
    }

    if (gradient !== 0) {
      const updateDirection = math.multiply(gradient, 1 / math.norm(gradient))
      this.policy.updateWeights(updateDirection)
    }
  }

  getSampleGradient({ state, action, advantage, actionProbability }) {
    const importanceWeight =
      this.policy.getProbability(state, action) / actionProbability

    return this.shouldClip(importanceWeight)
      ? 0
      : math.multiply(
          this.policy.partial(state, action),
          1 / actionProbability,
          advantage
        )
  }

  shouldClip() {
    // TODO
    return false
  }

  computeReturns() {
    const rewards = this.history.map(({ reward }) => reward)
    rewards.forEach((reward, t) => {
      this.history[t].return = this.getDiscountedReturn(rewards.slice(t))
    })
  }

  getBatches() {
    // TODO
  }

  getDiscountedReturn(rewards) {
    const discountedRewards = rewards.map(
      (reward, index) => Math.pow(this.getGamma(), index) * reward
    )
    return math.add(0, ...discountedRewards) // mathjs gets mad if you only pass one value
  }

  computeBaselines() {
    this.history.forEach(step => (step.baseline = this.v.call(step.state)))
  }

  updateBaselines(errors) {
    this.history.forEach(({ state }, i) => this.v.update(state, errors[i]))
  }

  getGamma() {
    return 1
  }
}
