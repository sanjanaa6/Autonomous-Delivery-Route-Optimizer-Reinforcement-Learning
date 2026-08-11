import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Any

class CityMapGenerator:
    """
    Generates city road network graphs, traffic conditions, and delivery order scenarios.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def create_grid_city(
        self, 
        grid_size: int = 4, 
        num_customers: int = 5, 
        num_chargers: int = 2,
        high_traffic_prob: float = 0.3
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Creates a grid-based city layout with depot at node 0, charging stations, 
        customer delivery locations, and variable edge traffic multipliers.
        """
        G = nx.grid_2d_graph(grid_size, grid_size)
        # Relabel nodes to 0..N-1 integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        num_nodes = G.number_of_nodes()
        
        # Node positions
        pos = {}
        for original_node, new_id in mapping.items():
            pos[new_id] = (float(original_node[0] * 5.0), float(original_node[1] * 5.0))
        
        nx.set_node_attributes(G, pos, 'pos')

        # Node types
        node_types = {}
        node_types[0] = 'depot'
        
        available_nodes = list(range(1, num_nodes))
        random.shuffle(available_nodes)
        
        chargers = available_nodes[:num_chargers]
        for c in chargers:
            node_types[c] = 'recharge'
            
        remaining_nodes = available_nodes[num_chargers:]
        customer_nodes = remaining_nodes[:min(num_customers, len(remaining_nodes))]
        for cust in customer_nodes:
            node_types[cust] = 'customer'
            
        for n in range(num_nodes):
            if n not in node_types:
                node_types[n] = 'junction'
                
        nx.set_node_attributes(G, node_types, 'type')

        # Edge base distances and initial traffic congestion factors
        traffic_dict = {}
        distance_dict = {}
        for u, v in G.edges():
            p1 = pos[u]
            p2 = pos[v]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distance_dict[(u, v)] = float(dist)
            distance_dict[(v, u)] = float(dist)

            # Random traffic congestion factor between 1.0 (free flow) and 2.5 (heavy traffic)
            if random.random() < high_traffic_prob:
                traffic = round(random.uniform(1.8, 2.8), 2)
            else:
                traffic = round(random.uniform(1.0, 1.3), 2)
            traffic_dict[(u, v)] = traffic
            traffic_dict[(v, u)] = traffic

        nx.set_edge_attributes(G, distance_dict, 'base_distance')
        nx.set_edge_attributes(G, traffic_dict, 'traffic_factor')

        # Generate Delivery Orders
        orders = self.generate_orders(customer_nodes, num_orders=num_customers)

        scenario_meta = {
            "depot_node": 0,
            "customer_nodes": customer_nodes,
            "charger_nodes": chargers,
            "grid_size": grid_size,
            "num_nodes": num_nodes,
            "orders": orders
        }

        return G, scenario_meta

    def generate_orders(self, customer_nodes: List[int], num_orders: int) -> List[Dict[str, Any]]:
        """
        Generates realistic delivery orders with weights, priority levels (High=3, Med=2, Low=1),
        and deadlines.
        """
        orders = []
        priorities = [3, 2, 1]  # High, Medium, Low
        weights = [3.0, 5.0, 8.0, 10.0, 12.0]

        for i in range(num_orders):
            target_node = customer_nodes[i % len(customer_nodes)]
            priority = random.choice(priorities)
            weight = random.choice(weights)
            # Deadline in simulation time steps (High priority has tighter deadline)
            deadline = random.randint(18, 40) if priority == 3 else random.randint(30, 60)
            
            orders.append({
                "order_id": i + 1,
                "node": target_node,
                "priority": priority,
                "weight": weight,
                "deadline": deadline,
                "status": "pending",  # 'pending', 'delivered', 'failed'
                "delivery_time": None
            })
        return orders

    def step_traffic_simulation(self, G: nx.Graph, fluctuation_rate: float = 0.2) -> nx.Graph:
        """
        Simulates dynamic traffic updates across the city network for a single timestep.
        """
        for u, v in G.edges():
            if random.random() < fluctuation_rate:
                current_tf = G[u][v].get('traffic_factor', 1.0)
                # Random shift in traffic factor [-0.3, +0.3], bounded between 1.0 and 3.0
                delta = round(random.uniform(-0.4, 0.4), 2)
                new_tf = max(1.0, min(3.0, round(current_tf + delta, 2)))
                G[u][v]['traffic_factor'] = new_tf
        return G
