import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from movingdot import MovingDotEnv
from ACAgent import ActorCriticAgent

patience = 10

hf_best_actor_path = "weights/hf_actor_best"
hf_best_critic_path = "weights/hf_critic_best"
hf_actor_converge_path = "weights/hf_actor_converge"
hf_critic_converge_path = "weights/hf_critic_converge"


def simulate(display, agent, env):
    state = env.reset()
    max_steps = 150

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

def simulate_human(display, agent, env):
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
        neighbor = [np.array([4.0, 5.0]),np.array([6.0, 5.0]),np.array([5.0, 4.0]),
                np.array([5.0, 6.0]),np.array([4.0, 4.0]),np.array([6.0, 4.0]),
                np.array([6.0, 6.0]),np.array([4.0, 6.0])]
        for _ in range(max_steps):
            flag = False
            for arr in neighbor:
                if np.array_equal(state, arr):
                    flag = True

            if flag:
                action = agent.choose_action_human(state)
            else:
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
    agent = ActorCriticAgent(state_size, action_size)

    best_reward = float('-inf')
    prev_reward = 0
    sum = 0
    cont_pos = 0

    for iteration in range(10000):
        env.reset()

        if iteration < 20 and iteration % 2 == 0:
            rewards, losses = simulate_human(False, agent, env)
        else:
            rewards, losses = simulate(False, agent, env)

        if rewards > 0:
            cont_pos += 1
        else:
            cont_pos = 0

        if cont_pos > 10:
            break

        if best_reward < rewards:
            best_reward = rewards
            print("saving...")
            agent.save(hf_best_actor_path, hf_best_critic_path)


        if abs(rewards - prev_reward) < 5:
            sum += 1
        else:
            sum = 0

        if sum >= patience or iteration > 200:
            break

        prev_reward = rewards

        print(iteration, "total rewards =", rewards, "| average loss =", losses[0])

    if(cont_pos > 10):
        agent.save(hf_actor_converge_path, hf_critic_converge_path)
    print("finished")

if __name__ == "__main__":
    main()