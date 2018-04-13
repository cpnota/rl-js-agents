const TRPO = require('./')

test('gets returns', () => {
  const agent = new TRPO({})
  agent.history = [1, 2, 3].map(reward => ({
    reward
  }))
  agent.computeReturns()
  expect(agent.history.map(({ return: r }) => r)).toEqual([6, 5, 3])
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
  agent.computeBaselines()
  expect(agent.history.map(({ baseline }) => baseline)).toEqual([2, 4, 6])
})

test('computes advantages', () => {
  const agent = new TRPO({
    v: {
      call: state => state * 2
    }
  })
  agent.history = [1, 2, 3].map(state => ({
    state,
    reward: state
  }))
  agent.computeAdvantages()
  expect(agent.history.map(({ advantage }) => advantage)).toEqual([4, 1, -3])
})
