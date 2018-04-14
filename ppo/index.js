const math = require('mathjs')
const MiniBatchSGD = require('./optimize/minibatch-sgd')

module.exports = class ProximalPolicyOptimization {
  constructor({ policy, v, epsilon, batchStrategy, optimizer }) {
    this.policy = policy
    this.v = v
    this.epsilon = epsilon
    this.batchStrategy = batchStrategy
    this.history = []
    this.optimizer = optimizer
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

    if (this.batchStrategy.shouldUpdate(this.history)) {
      this.update()
      this.history = []
    }
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
    const samples = this.history
    const computeGradient = sample => this.getSampleGradient(sample)
    const update = direction => this.policy.updateWeights(direction)
    this.optimizer.optimize({
      samples,
      computeGradient,
      update
    })
  }

  computeBaselines() {
    this.history.forEach(step => (step.baseline = this.v.call(step.state)))
  }

  updateBaselines(errors) {
    this.history.forEach(({ state }, i) => this.v.update(state, errors[i]))
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

  getGamma() {
    return 1
  }

  getLambda() {
    return 1
  }
}
