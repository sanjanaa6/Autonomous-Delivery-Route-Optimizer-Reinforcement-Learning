from typing import Dict, List, Any

class XAIRouteExplainer:
    """
    Explainable AI (XAI) Reasoning Engine for Transparent Autonomous Route Recommendations.
    Translates complex Deep Q-Network reward values into clear, human-understandable natural language explanations.
    """
    def __init__(self):
        pass

    def explain_selection(
        self,
        env,
        selected_route_idx: int,
        all_rewards: Dict[int, float],
        all_breakdowns: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        routes = env.routes
        if selected_route_idx >= len(routes):
            return {
                "headline": "Invalid Route Selected",
                "explanation": "Agent selected an out-of-bounds route index.",
                "tradeoff_points": []
            }

        chosen = routes[selected_route_idx]
        chosen_reward = all_rewards[selected_route_idx]
        chosen_bd = all_breakdowns[selected_route_idx]

        # Identify alternatives
        alternatives = [
            (idx, r, all_rewards[idx], all_breakdowns[idx])
            for idx, r in enumerate(routes)
            if idx != selected_route_idx
        ]

        headline = f"Selected {chosen['name']} (Efficiency Reward: {chosen_reward:.1f})"

        tradeoff_points = []
        key_reasons = []

        for alt_idx, alt_route, alt_reward, alt_bd in alternatives:
            reward_diff = chosen_reward - alt_reward
            time_diff = alt_route["duration_min"] - chosen["duration_min"]  # positive if chosen is faster
            dist_diff = chosen["distance_km"] - alt_route["distance_km"]  # positive if chosen is longer
            traffic_diff = alt_route["traffic_factor"] - chosen["traffic_factor"] # positive if chosen has less traffic
            toll_diff = alt_route["toll_cost"] - chosen["toll_cost"] # positive if chosen is cheaper

            point_desc = []

            if time_diff > 1.0:
                point_desc.append(f"saves {abs(time_diff):.1f} mins of travel time")
            elif time_diff < -1.0:
                point_desc.append(f"adds {abs(time_diff):.1f} mins of travel time")

            if traffic_diff > 0.2:
                point_desc.append(f"bypasses heavy traffic congestion ({chosen['traffic_factor']:.2f}x vs {alt_route['traffic_factor']:.2f}x)")
            elif traffic_diff < -0.2:
                point_desc.append(f"has slightly higher traffic congestion ({chosen['traffic_factor']:.2f}x vs {alt_route['traffic_factor']:.2f}x)")

            if toll_diff > 0.5:
                point_desc.append(f"saves ₹{toll_diff:.2f} in toll fees")
            elif toll_diff < -0.5:
                point_desc.append(f"incurs ₹{abs(toll_diff):.2f} in tolls")

            if dist_diff > 1.0:
                point_desc.append(f"covers {abs(dist_diff):.1f} km extra distance")
            elif dist_diff < -1.0:
                point_desc.append(f"is {abs(dist_diff):.1f} km shorter")

            reason_str = f"Compared to **{alt_route['name']}** (Reward: {alt_reward:.1f}), the selected route " + ", and ".join(point_desc) + "."
            tradeoff_points.append(reason_str)

        # Primary natural language summary paragraph
        summary_parts = [
            f"The AI Agent recommended **{chosen['name']}** as the optimal route based on multi-objective Reinforcement Learning optimization."
        ]

        if any(alt_route["traffic_factor"] > 1.7 for _, alt_route, _, _ in alternatives) and chosen["traffic_factor"] <= 1.3:
            summary_parts.append("It successfully identified and avoided major traffic bottlenecks along alternative corridors.")

        if chosen["toll_cost"] == 0.0 and any(alt_route["toll_cost"] > 0 for _, alt_route, _, _ in alternatives):
            summary_parts.append("It prioritized a toll-free path without sacrificing significant delivery time.")

        if env.priority == 3:
            summary_parts.append("Given the **High Priority** delivery requirement, time savings were heavily weighted over distance.")

        explanation_text = " ".join(summary_parts)

        return {
            "headline": headline,
            "explanation": explanation_text,
            "tradeoff_points": tradeoff_points,
            "chosen_route": chosen,
            "chosen_reward": chosen_reward,
            "reward_breakdown": chosen_bd
        }
