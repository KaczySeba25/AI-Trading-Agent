import asyncio
import numpy as np

from live_data import LiveDataFeed
from feature_extractor_88 import FeatureExtractor88
from replay_buffer import ReplayBuffer
from environment_features import FeatureEnvironment
from agent_ppo_88 import PPOAgent

async def main():
    print("Ładuję miesiąc historii + uruchamiam LIVE feed...")

    feed = LiveDataFeed()
    fe = FeatureExtractor88()
    buffer = ReplayBuffer(capacity=50000)

    asyncio.create_task(feed.connect())

    print(f"Organizm startuje z {len(feed.df)} świecami (miesiąc historii).")

    env = FeatureEnvironment(fe)
    agent = PPOAgent(env)

    state, _ = env.reset()

    step_count = 0

    print("Organizm działa LIVE... Uczy się na każdym tiknięciu...")

    while True:
        df = feed.get_history()

        features = fe.compute(df)
        env.last_features = features

        action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        buffer.push(state, action, reward, next_state, terminated)

        state = next_state
        step_count += 1

        if step_count % 50 == 0:
            print(f"[TRAIN] Mini‑training PPO... krok: {step_count}")
            agent.train(steps=2048)
            agent.save("models/ppo_88")

        if terminated:
            state, _ = env.reset()

        await asyncio.sleep(1)

asyncio.run(main())
