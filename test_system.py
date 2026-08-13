import sys

def run_tests():
    print("Testing GoogleMapsRouteClient...")
    from google_maps.api_client import GoogleMapsRouteClient
    client = GoogleMapsRouteClient()
    scen = client.fetch_routes("New York, NY", "Boston, MA")
    assert len(scen["routes"]) >= 2, "Expected at least 2 candidate routes"
    print("[OK] GoogleMapsRouteClient OK")

    print("Testing RealWorldRouteEnv...")
    from rl_core.route_env import RealWorldRouteEnv
    env = RealWorldRouteEnv(scen)
    obs, info = env.reset()
    assert obs.shape[0] > 0, "Observation space invalid"
    next_obs, reward, term, trunc, info = env.step(0)
    assert isinstance(reward, float), "Reward must be float"
    print("[OK] RealWorldRouteEnv OK")

    print("Testing PyTorch DQNAgent & Trainer...")
    from rl_core.trainer import train_dqn_route_agent, evaluate_route_policy
    agent, df_hist = train_dqn_route_agent(env, num_episodes=5)
    assert len(df_hist) == 5, "Expected 5 training episodes"
    print("[OK] PyTorch DQNAgent & Trainer OK")

    print("Testing XAIRouteExplainer Engine...")
    eval_res = evaluate_route_policy(env, agent, solver_type="dqn")
    xai = eval_res["explanation"]
    assert "headline" in xai, "XAI output missing headline"
    assert "explanation" in xai, "XAI output missing explanation text"
    print("[OK] XAIRouteExplainer Engine OK")

    print("Testing Classical Route Solvers...")
    from baselines.solvers import ShortestDistanceSolver, FastestDurationSolver, LowestCostSolver
    s1 = ShortestDistanceSolver()
    s2 = FastestDurationSolver()
    s3 = LowestCostSolver()

    a1 = s1.select_action(env)
    a2 = s2.select_action(env)
    a3 = s3.select_action(env)

    assert 0 <= a1 < len(env.routes)
    assert 0 <= a2 < len(env.routes)
    assert 0 <= a3 < len(env.routes)
    print("[OK] Classical Route Solvers OK")

    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
