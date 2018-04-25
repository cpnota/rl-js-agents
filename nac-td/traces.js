const math = require('mathjs')

export default class AdvantageTraces {
  constructor (advantage) {
    this.advantage = advantage
    this.traces = 0 // lazy initialize
  }

  record(state, action) {
    const partial = this.advantage.policy.partialLN(this.state, this.action)
    this.traces = math.add(this.traces, partial)
  }

  updateAdvantage(error) {
    this.advantage.updateWeights(this.traces, error)
  }

  decay(amount) {
    this.traces = math.multiply(amount, this.traces)
  }
}
