import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from movingdot import MovingDotEnv
from ACAgent import ActorCriticAgent

patience = 10


def simulate(display, agent, env):
    state = env.reset()
    max_steps = 100

    def update(frame):
        nonlocal state
        ax.clear()

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)

        if not done:
            state = next_state
            env.render(ax)
            plt.pause(0.1)
        else:
            env.render(ax)
            plt.pause(0.1)
            ani.event_source.stop()
            plt.close()
            return

    def run():
        nonlocal state
        rewards = 0
        losses = []
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            losses.append(agent.learn(state, action, reward, next_state, done))

            rewards += reward

            if not done:
                state = next_state
            else:
                break

        return rewards, (sum(losses) / len(losses))

    if display:
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=max_steps, repeat=False)
        plt.show()
    else:
        return run()




def main():
    env = MovingDotEnv()
    state_size = np.prod(np.array(env.observation_space.shape))
    action_size = env.action_space.n
    count = 0
    rewards = float('-inf')
    best_reward = float('-inf')

    while(rewards < 0):

        sum = 0
        prev_reward = 0
        reach_max = False
        agent = ActorCriticAgent(state_size, action_size)
        cont_pos = 0
        print(f"\nStart simulation {count}")

        for iteration in range(10000):
            env.reset()

            rewards, losses = simulate(False, agent, env)

            if best_reward < rewards:
                best_reward = rewards
                print("saving...")
                agent.save("weights/actor_weights_best", "weights/critic_weights_best")

            if iteration > 50:

                if rewards > 0:
                    cont_pos += 1
                else:
                    cont_pos = 0

                if cont_pos > 10:
                    break

                if abs(rewards - prev_reward) < 5:
                    sum += 1
                else:
                    sum = 0

                if sum >= patience:
                    break

                if iteration > 500:
                    reach_max = True
                    break


                prev_reward = rewards



            print(iteration, "total rewards =", rewards, "| average loss =", losses[0])

        print(f'\nrewards converge to {rewards} in simulation {count}')
        count += 1

    if reach_max == False:
        agent.save("weights/actor_weights_converge","weights/critic_weights_converge")

    print(f"\nfinished   Total simulation {count}")

if __name__ == "__main__":
    main()