// mock to allow instansiation of interfaces directly as mocks
const {
  Policy,
  ActionValueFunction,
  ActionTraces
} = require('@rl-js/interfaces')

// walk the prototype chain to get all
// methods required for interface
const getAllRequiredMethods = obj => {
  if (obj.name === '') return []
  return Object.getOwnPropertyNames(obj.prototype).concat(
    getAllRequiredMethods(Object.getPrototypeOf(obj))
  )
}

const mock = Interface => {
  class Impl extends Interface {}
  const requiredMethods = getAllRequiredMethods(Interface)
  requiredMethods.forEach(method => {
    if (method !== 'constructor') {
      Impl.prototype[method] = jest.fn()
    }
  })
  return Impl
}

module.exports = {
  Policy: mock(Policy),
  ActionValueFunction: mock(ActionValueFunction),
  ActionTraces: mock(ActionTraces)
}
