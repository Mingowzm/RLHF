import torch
import torch.nn as nn
import torch.optim as optim

actor_weights_path = "weights/actor_weights.h5"
critic_weights_path = "weights/critic_weights.h5"

actor_weights_path_best = "weights/actor_weights.h1"
critic_weights_path_best = "weights/critic_weights.h1"


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.network(state)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate_actor=1e-4, learning_rate_critic=1e-4, gamma=0.90, beta=0.1):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        self.gamma = gamma
        self.beta = beta

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probabilities = self.actor(state)
        action = torch.multinomial(probabilities, 1).item()
        return action

    def choose_action_human(self, state):
        print(f"Current state is {state}\n")
        print("Waht is the next action?\n")
        print("Up:0  Down:1  Left:2  Right:3\n")
        user_input = input("Please choose: ")
        return int(user_input)

    def learn(self, state, action, reward, next_state, done):
        #self.actor.train()
       #self.critic.train()

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target - value
        self.optimizer_critic.zero_grad()
        td_error.backward()
        self.optimizer_critic.step()

        # Actor update
        probabilities = self.actor(state)
        distribution = torch.distributions.Categorical(probabilities)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        loss_actor = -(log_prob * td_error.detach() + self.beta * entropy)
        '''If an action results in a higher reward than expected (positive TD error),
           this term will be positive, and the gradient ascent will increase the probability
           of taking that action in the future. Conversely, if the action was worse than
           expected (negative TD error), the term will be negative, and the probability of
           taking that action will decrease.'''
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        return loss_actor

    def save(self, actor_weights_path, critic_weights_path):
        torch.save(self.actor.state_dict(), actor_weights_path)
        torch.save(self.critic.state_dict(), critic_weights_path)

    def load(self,actor_weights_path, critic_weights_path):
        self.actor.load_state_dict(torch.load(actor_weights_path))
        self.critic.load_state_dict(torch.load(critic_weights_path))

    def load_best(self):
        self.actor.load_state_dict(torch.load(actor_weights_path_best))
        self.critic.load_state_dict(torch.load(critic_weights_path_best))




