const Reinforce = require('./index')

test('gets the returns at each timestep', () => {
  const reinforce = new Reinforce({})
  reinforce.environment = {
    gamma: 0.5
  }
  reinforce.history = [1, 2, 4, 8].map(reward => ({ reward }))
  const returns = reinforce.getReturns()
  expect(returns).toEqual([4, 6, 8, 8])
})

test('does the policy update', () => {
  const policy = {
    update: jest.fn()
  }

  const reinforce = new Reinforce({ policy, alphaBaseline: 2 })

  reinforce.environment = {
    gamma: 0.5
  }

  reinforce.baseline = 2

  reinforce.history = [1, 2, 4, 8].map(reward => ({
    state: reward,
    action: reward,
    reward
  }))

  reinforce.update()

  expect(policy.update.mock.calls).toEqual([
    [1, 1, 2],
    [2, 2, 4],
    [4, 4, 6],
    [8, 8, 6]
  ])

  expect(reinforce.baseline).toEqual(11)
})
