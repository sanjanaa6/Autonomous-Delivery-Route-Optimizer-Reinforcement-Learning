import sys
import numpy as np

def run_tests():
    print("Testing CityMapGenerator...")
    from utils.map_generator import CityMapGenerator
    map_gen = CityMapGenerator(seed=42)
    G, meta = map_gen.create_grid_city(grid_size=4, num_customers=5, num_chargers=2)
    assert G.number_of_nodes() == 16, f"Expected 16 nodes, got {G.number_of_nodes()}"
    assert len(meta["orders"]) == 5, f"Expected 5 orders, got {len(meta['orders'])}"
    print("[OK] CityMapGenerator OK")

    print("Testing DeliveryEnv...")
    from rl_core.env import DeliveryEnv
    env = DeliveryEnv(G, meta, max_steps=50)
    obs, info = env.reset()
    assert obs.shape[0] > 0, "Observation shape invalid"
    assert info["current_node"] == 0, "Initial node should be depot (0)"
    
    # Step random action
    next_obs, reward, term, trunc, info = env.step(1)
    assert isinstance(reward, float), "Reward must be float"
    print("[OK] DeliveryEnv OK")

    print("Testing Baseline Solvers...")
    from baselines.solvers import DijkstraSolver, GreedyTSPSolver, DynamicTrafficDijkstraSolver
    dijkstra = DijkstraSolver()
    greedy = GreedyTSPSolver()
    dyn_dijk = DynamicTrafficDijkstraSolver()

    a1 = dijkstra.select_action(env)
    a2 = greedy.select_action(env)
    a3 = dyn_dijk.select_action(env)
    assert 0 <= a1 <= env.num_nodes
    assert 0 <= a2 <= env.num_nodes
    assert 0 <= a3 <= env.num_nodes
    print("[OK] Baseline Solvers OK")

    print("Testing Tabular Q-Learning Agent...")
    from rl_core.trainer import train_q_learning
    q_agent, q_df = train_q_learning(env, num_episodes=5)
    assert len(q_df) == 5, "Expected 5 episode records in q_df"
    print("[OK] Q-Learning Agent OK")

    print("Testing PyTorch DQN Agent...")
    from rl_core.trainer import train_dqn
    dqn_agent, dqn_df = train_dqn(env, num_episodes=5)
    assert len(dqn_df) == 5, "Expected 5 episode records in dqn_df"
    print("[OK] PyTorch DQN Agent OK")

    print("Testing Benchmark Pipeline...")
    from utils.metrics import run_comprehensive_benchmark
    solvers = {"dijkstra": dijkstra, "greedy": greedy, "dynamic_dijkstra": dyn_dijk}
    df_bench, _ = run_comprehensive_benchmark(env, q_agent=q_agent, dqn_agent=dqn_agent, solvers_dict=solvers, num_episodes=2)
    assert len(df_bench) == 5, f"Expected 5 benchmark rows, got {len(df_bench)}"
    print("[OK] Benchmark Pipeline OK")

    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
