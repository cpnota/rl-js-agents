const math = require('mathjs')

// TODO:
// Much better performance on CartPole was achieved with 
// a once-per-episode update strategy
// with no mini-batches.
// The update strategy as well as the SGD
// algorithms should be strategies
module.exports = class ProximalPolicyOptimization {
  constructor({ policy, v, epochs, epsilon, batchSize, miniBatchSize = 50 }) {
    this.policy = policy
    this.v = v
    this.epochs = epochs
    this.epsilon = epsilon
    this.batchSize = batchSize
    this.miniBatchSize = miniBatchSize
    this.history = []
  }

  newEpisode(environment) {
    this.environment = environment
  }

  act() {
    const state = this.environment.getState()
    const action = this.policy.chooseAction(state)
    const actionProbability = this.policy.getProbability(state, action)
    this.environment.dispatch(action)
    const reward = this.environment.getReward()
    const terminal = this.environment.isTerminated()
    this.history.push({ state, action, reward, actionProbability, terminal })

    if (this.shouldUpdate()) {
      this.update()
      this.history = []
    }
  }

  shouldUpdate() {
    return this.history.length >= this.batchSize
  }

  update() {
    this.updateValueFunction()
    this.computeAdvantages()
    this.optimizePolicy()
  }

  updateValueFunction() {
    this.history.forEach(({ state, reward, terminal }, t) => {
      const estimate = this.v.call(state)

      if (!terminal && !this.history[t + 1]) {
        return // cannot compute tdError
      }

      const nextEstimate = terminal ? 0 : this.v.call(this.history[t + 1].state)
      const tdError = reward + this.getGamma() * nextEstimate - estimate
      this.v.update(state, tdError)
    })
  }

  computeAdvantages() {
    this.computeBaselines()
    this.computeReturns()
    this.computeTdErrors()

    const discountRate = this.getGamma() * this.getLambda()

    // TODO optimize this loop
    // it should be able to do this faster than n ** 2
    for (const t in this.history) {
      const step = this.history[t]
      const rest = this.history.slice(t)
      let advantage = 0
      for (const x in rest) {
        const { tdError = 0 } = rest[x]
        advantage += tdError * discountRate ** (x - t)
      }
      step.advantage = advantage
    }
  }

  optimizePolicy() {
    let count = 0
    let stepIndex = 0
    while (count < this.batchSize * this.epochs) {
      let gradient = 0
      for (let i = 0; i < this.miniBatchSize; i++) {
        const step = this.history[stepIndex]
        gradient = math.add(gradient, this.getSampleGradient(step))
        count += 1
        stepIndex += 1
        if (!this.history[stepIndex]) stepIndex = 0
      }
      if (gradient === 0) {
        // we've either completely converged,
        // or there was some error.
        return
      }
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

  shouldClip(importanceWeight) {
    return Math.abs(1 - importanceWeight) >= this.epsilon
  }

  computeTdErrors() {
    this.history.forEach((step, t) => {
      const { state, reward, terminal } = step
      const estimate = this.v.call(state)

      if (!terminal && !this.history[t + 1]) {
        return // cannot compute tdError
      }

      const nextEstimate = terminal ? 0 : this.v.call(this.history[t + 1].state)
      step.tdError = reward + this.getGamma() * nextEstimate - estimate
    })
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

  getLambda() {
    return 1
  }
}
