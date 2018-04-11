const TRPO = require('./')

test('gets returns', () => {
  const agent = new TRPO({})
  agent.history = [1, 2, 3].map(reward => ({
    reward
  }))
  expect(agent.getReturns()).toEqual([6, 5, 3])
})

test('gets baselines', () => {
  const agent = new TRPO({
    v: {
      call: state => state * 2
    }
  })
  agent.history = [1, 2, 3].map(state => ({
    state
  }))
  expect(agent.getBaselines()).toEqual([2, 4, 6])
})

test('computes policy gradient', () => {
  const agent = new TRPO({
    policy: {
      partialLN: (state, action) => [state, action]
    }
  })
  const errors = [1, 0, -1]
  agent.history = [1, 2, 3].map(state => ({
    state,
    action: state / 2
  }))
  expect(agent.policyGradient(errors)).toEqual([-2, -1])
})
