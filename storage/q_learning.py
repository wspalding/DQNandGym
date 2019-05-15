import numpy as np
import gym

class Q_model():
    def __init__(self, env, **kwargs):
        self.env = env
        assert(self.env)
        self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        self.eta = kwargs.get("eta", 0.628)
        self.gamma = kwargs.get("gamma", .9)
        self.episodes = kwargs.get("episodes", 5000)

        #self.reward_list = []
        self.train()
    
    def train(self):
        for i in range(0, self.episodes):
            obs = self.env.reset()
            j = 0
            while j < 99:
                #env.render()
                j+=1
                # Choose action from Q table
                a = self.get_action(obs)
                #Get new state & reward from environment
                obs2,r,d,_ = self.env.step(a)
                #Update Q-Table with new knowledge
                self.update_table(obs, obs2, r, a)
                
                obs = obs2
                if d == True:
                    break

    

    def get_action(self, observation):
        return np.argmax(self.Q_table[observation,:] + np.random.randn(1,self.env.action_space.n)*(1./(self.episodes+1)))

    def update_table(self, obs0, obs1, reward, action):
        self.Q_table[obs0,action] = self.Q_table[obs0,action] + self.eta*(reward + self.gamma*np.max(self.Q_table[obs1,:]) - self.Q_table[obs0,action])
