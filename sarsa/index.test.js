const Sarsa = require('./index')

describe('sarsa', () => {
  let environment
  let q
  let policy
  let agent

  const state1 = 'state1'
  const state2 = 'state2'
  const action1 = 'action1'
  const action2 = 'action2'

  beforeEach(() => {
    environment = {
      getState: jest.fn(),
      dispatch: jest.fn(),
      isTerminated: jest.fn(),
      getReward: jest.fn(),
      gamma: 0.5
    }

    q = {
      call: jest.fn(),
      update: jest.fn()
    }

    policy = {
      chooseAction: jest.fn()
    }

    agent = new Sarsa({ q, policy })

    environment.getState.mockReturnValueOnce(state1)
    policy.chooseAction.mockReturnValueOnce(action1)
    agent.newEpisode(environment)

    environment.getState.mockReturnValueOnce(state2)
    policy.chooseAction.mockReturnValueOnce(action2)
    environment.getReward.mockReturnValueOnce(3)
    q.call.mockReturnValueOnce(1).mockReturnValueOnce(2)
  })

  test('chooses an action when episode is initialized', () => {
    expect(policy.chooseAction).toBeCalledWith(state1)
  })

  test('dispatches the first action', () => {
    agent.act()
    expect(environment.dispatch).toBeCalledWith('action1')
  })

  test('chooses the next action', () => {
    agent.act()
    expect(policy.chooseAction).toBeCalledWith(state2)
  })

  test('updates q according to the TD error', () => {
    agent.act()
    expect(q.update).toBeCalledWith(state1, action1, 1.5)
  })

  test('dispatches the next action on second time step', () => {
    agent.act()
    agent.act()
    expect(environment.dispatch).toBeCalledWith('action2')
  })

  test('updates q on second time step with second state and action', () => {
    agent.act()
    q.call.mockReturnValueOnce(10).mockReturnValueOnce(20)
    environment.getReward.mockReturnValueOnce(5)
    agent.act()
    expect(q.update).lastCalledWith(state2, action2, -10)
  })

  test('if environment is terminated, uses 0 as next estimate in TD error calculation', () => {
    environment.isTerminated.mockReturnValueOnce(true)
    agent.act()
    expect(q.update).toBeCalledWith(state1, action1, 2)
  })
})
