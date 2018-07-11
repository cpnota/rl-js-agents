const math = require('mathjs')
const Agent = require('@rl-js/interfaces/agent')

module.exports = class Reinforce extends Agent {
  constructor({ policy, alphaBaseline }) {
    super()
    this.policy = policy
    this.alphaBaseline = alphaBaseline
    this.baseline = 0
  }

  newEpisode(environment) {
    this.environment = environment
    this.history = []
  }

  act() {
    const state = this.environment.getState()
    const action = this.policy.chooseAction(state)
    this.environment.dispatch(action)
    const reward = this.environment.getReward()
    this.history.push({ state, action, reward })

    if (this.environment.isTerminated()) {
      this.update()
    }
  }

  update() {
    const returns = this.getReturns()
    returns.map(r => r - this.baseline).forEach((error, time) => {
      const { state, action } = this.history[time]
      this.policy.update(state, action, error)
    })
    this.baseline += this.alphaBaseline * (math.mean(returns) - this.baseline)
  }

  getReturns() {
    const rewards = this.history.map(({ reward }) => reward)
    return rewards.map((reward, index) =>
      this.getDiscountedReturn(rewards.slice(index))
    )
  }

  getDiscountedReturn(rewards) {
    const discountedRewards = rewards.map(
      (reward, index) => Math.pow(this.environment.gamma, index) * reward
    )
    return math.add(0, ...discountedRewards) // mathjs gets mad if you only pass one value
  }
}
