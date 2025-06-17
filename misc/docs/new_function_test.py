# /*****************************************************************************/
#  * File: forest_pruner.py
#  * Author: Olavo Alves Barros Silva (com contribuições de IA)
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-06-16
#  * License: [License Type]
#  * Description: Implements strategic forest pruning with advanced probability
#  * calculation and simulation.
#  *****************************************************************************/

import math
import random
from collections import Counter

# Assume que a classe TreePruningHash está definida em outro lugar e importada
# from tree_pruning import TreePruningHash
class TreePruningHash:
    # Esta é a classe de treePruningHash.py.
    # Coloquei aqui para o código ser executável, mas ela pode ser importada.
    def should_cut_node(self, node_id, cut_probability, salt=0):
        if node_id == 0: return 0
        if cut_probability <= 0: return 0
        if cut_probability >= 1: return 1
        hash_value = self._generate_hash_value(node_id, salt)
        threshold = int(cut_probability * 2147483647.0)
        return 1 if (hash_value & 0x7FFFFFFF) < threshold else 0

    def _generate_hash_value(self, node_id, prob_index=0):
        combined = (node_id << 32) | (prob_index & 0xFFFFFFFF)
        state = (1103515245 * combined + 12345) & 0xFFFFFFFF
        return state

class ForestPruner:
    def __init__(self, forest_trees):
        """
        Initializes the pruner with a list of trees.
        
        Args:
            forest_trees (list): A list where each element is a dict representing a tree.
        """
        self._trees = forest_trees
        self._prune_probabilities = None
        self._total_nodes = sum(len(tree) for tree in self._trees)

    def _get_node_levels(self, tree: dict) -> dict:
        """
        [NEW] Private function to get the level of each node in a single tree.
        This fulfills your request #1.
        
        Args:
            tree (dict): The tree structure.
            
        Returns:
            dict: A mapping of {node_id: level}.
        """
        levels = {}
        for node_id in tree.keys():
            level = 0
            curr_id = node_id
            # Traverse up the tree to find the depth
            while 'parent_node' in tree.get(curr_id, {}):
                level += 1
                curr_id = tree[curr_id]['parent_node']
            levels[node_id] = level
        return levels

    def _simulate_cut_decision(self, pruner: TreePruningHash, prob: float, global_id: int, tree_index: int) -> int:
        """
        [NEW] Simulates the cut decision across 1000 threads and returns the mode.
        This fulfills your request #3, based on your original code.
        
        Args:
            pruner (TreePruningHash): The hashing object.
            prob (float): The cut probability.
            global_id (int): The global ID of the node, used as a salt.
            tree_index (int): The index of the tree, for an extra salt.
            
        Returns:
            int: 1 if the majority of threads decided to cut, 0 otherwise.
        """
        if prob == 0: return 0
        if prob == 1: return 1
        
        decisions = []
        # Simulate with 1000 different thread IDs for robustness
        for thread_id in range(1000):
            # The salt combines global_id and tree_index to ensure determinism
            # is unique for each node across the entire forest.
            salt = global_id + tree_index 
            decision = pruner.should_cut_node(node_id=thread_id, cut_probability=prob, salt=salt)
            decisions.append(decision)
        
        # Count votes and return the majority decision
        counts = Counter(decisions)
        return counts.most_common(1)[0][0]

    def _calculate_cut_probability(self, node_level, nodes_cut_so_far, total_nodes_to_cut,
                                   nodes_processed_so_far, total_nodes_in_forest,
                                   nodes_cut_in_tree, nodes_in_tree, max_cut_percentage,
                                   urgency_override_threshold, strategy,
                                   level_importance, progress_importance, level_bias):
        """
        [REFACTORED] Calculates the probability for a single node.
        All complexity is now contained here, fulfilling your request #2.
        The mathematical formula is P(cut) = min(1.0, P_base * L_factor * U_factor).
        """
        # --- Soft Per-Tree Limit Logic ---
        max_nodes_in_tree = int(nodes_in_tree * max_cut_percentage)
        is_tree_limit_reached = nodes_cut_in_tree >= max_nodes_in_tree

        remaining_to_process = total_nodes_in_forest - nodes_processed_so_far
        remaining_to_cut = total_nodes_to_cut - nodes_cut_so_far

        if remaining_to_process > 0:
            urgency_ratio = remaining_to_cut / remaining_to_process
            is_desperate = urgency_ratio > (percentage_to_cut * urgency_override_threshold)
            if is_tree_limit_reached and not is_desperate:
                return 0.0 # Respect the limit, as we are not desperate

        if remaining_to_cut <= 0:
            return 0.0 # Goal reached

        # --- Base Probability Calculation ---
        base_prob = urgency_ratio # The urgency ratio is the most accurate base probability

        # --- Factor Calculation ---
        level_factor = 1.0
        urgency_factor = 1.0 # This is for the 'adaptive' strategy

        if strategy == "linear":
            level_factor = level_bias + (node_level * level_importance)
        elif strategy == "exponential":
            level_factor = level_bias * ((1 + level_importance) ** node_level)
        elif strategy == "adaptive":
            level_factor = level_bias + (node_level * level_importance)
            progress_ratio = nodes_cut_so_far / total_nodes_to_cut if total_nodes_to_cut > 0 else 0
            urgency_factor = 1.0 + (0.5 - progress_ratio) * progress_importance
            urgency_factor = max(0.1, urgency_factor)

        final_prob = base_prob * level_factor * urgency_factor
        return min(1.0, max(0.0, final_prob))

    def calculate_forest_probabilities(self, percentage_to_cut, strategy="adaptive",
                                       level_importance=0.5, progress_importance=0.3,
                                       level_bias=2.0, max_cut_percentage=0.3,
                                       urgency_override_threshold=1.5):
        """
        [REFACTORED] A lean orchestrator function.
        It loops through nodes and calls helper functions for complex logic.
        """
        total_nodes_to_cut = int(self._total_nodes * percentage_to_cut)
        nodes_processed_so_far = 0
        nodes_cut_so_far = 0
        
        all_probabilities = {}
        pruner = TreePruningHash()

        for tree_index, tree in enumerate(self._trees):
            nodes_in_tree = len(tree)
            nodes_cut_in_tree = 0
            node_levels = self._get_node_levels(tree)

            for node_id, node in tree.items():
                prob = self._calculate_cut_probability(
                    node_level=node_levels[node_id],
                    nodes_cut_so_far=nodes_cut_so_far,
                    total_nodes_to_cut=total_nodes_to_cut,
                    nodes_processed_so_far=nodes_processed_so_far,
                    total_nodes_in_forest=self._total_nodes,
                    nodes_cut_in_tree=nodes_cut_in_tree,
                    nodes_in_tree=nodes_in_tree,
                    max_cut_percentage=max_cut_percentage,
                    urgency_override_threshold=urgency_override_threshold,
                    strategy=strategy,
                    level_importance=level_importance,
                    progress_importance=progress_importance,
                    level_bias=level_bias
                )
                
                global_id = node['global_id']
                all_probabilities[global_id] = prob
                
                if self._simulate_cut_decision(pruner, prob, global_id, tree_index):
                    nodes_cut_so_far += 1
                    nodes_cut_in_tree += 1
                
                nodes_processed_so_far += 1
        
        self._prune_probabilities = [all_probabilities[i] for i in sorted(all_probabilities.keys())]
        return self._prune_probabilities
