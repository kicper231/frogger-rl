import gymnasium as gym
from stable_baselines3 import PPO
import ale_py

gym.register_envs(ale_py)

# train_env = gym.make("ALE/Frogger-v5")
# model = PPO("CnnPolicy", train_env, verbose=1)
# model.learn(total_timesteps=400_000)
# model.save("ppo_frogger_model")
# train_env.close()
# model.save("models/ppo_frogger_model_4")
model = PPO.load("models/ppo_frogger_model_2.zip")

test_env = gym.make("ALE/Frogger-v5", render_mode="human")
obs, info = test_env.reset()
done = False


while not done:
    action = test_env.action_space.sample()
    # action, _ = model.predict(obs)
    # print(action)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    print(reward)
    # test_env.render()

obs, info = test_env.reset()
done = False

while not done:
    # action = test_env.action_space.sample()
    action, _ = model.predict(obs)
    # print(action)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    # test_env.render()

