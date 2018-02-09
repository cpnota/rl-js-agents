module.exports = class Simple {
  constructor({ policy }) {
    this.policy = policy
  }

  newEpisode(environment) {
    this.environment = environment
  }

  act() {
    const state = this.environment.getState()
    const action = this.policy.chooseAction(state)
    this.environment.dispatch(action)
  }
}
