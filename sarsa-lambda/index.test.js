const { ActionValueFunction, ActionTraces, Policy } = require('../__mocks__/')
const SarsaLambda = require('./')

describe('constructor', () => {
  test('constructs object', () => {
    expect(
      () =>
        new SarsaLambda({
          actionValueFunction: new ActionValueFunction(),
          actionTraces: new ActionTraces(),
          policy: new Policy(),
          lambda: 0.5
        })
    ).not.toThrow()
  })

  const argNames = [
    'actionValueFunction',
    'actionTraces',
    'policy',
    'lambda',
    'gamma'
  ]

  argNames.forEach(arg => {
    test(`throws error if ${arg} is not defined`, () => {
      const args = {
        actionValueFunction: new ActionValueFunction(),
        actionTraces: new ActionTraces(),
        policy: new Policy(),
        lambda: 0.5,
        gamma: 1
      }
      args[arg] = null
      expect(() => new SarsaLambda(args)).toThrow(TypeError)
    })
  })
})
