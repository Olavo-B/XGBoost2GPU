# /*****************************************************************************/
#  * File: xgboost2gpu.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-06-11
#  * License: [License Type]
#  * Description: This is the main class to generate the cuda code 
#  * for the XGBoost algorithm.
# /*****************************************************************************/

import os
import sys

import math
import random
import collections
import numpy as np
from treelut import TreeLUTClassifier

from .treePruningHash import TreePruningHash

class XGBoost2GPU:
    def __init__(self, treelut_model: TreeLUTClassifier,
                 w_feature: int = 3,
                 w_tree: int = 3,
                 n_samples: int = 1000,
                 n_threads: int = 256,
                 n_blocks: int = 768,
                 debug: bool = False
                 ):
        self._model             = treelut_model
        self._w_feature         = w_feature
        self._w_tree            = w_tree
        self._n_samples         = n_samples

        self._n_threads         = n_threads
        self._n_blocks          = n_blocks
        self._debug             = debug

        self._n_classes         = treelut_model.n_classes
        self._trees             = treelut_model.trees
        self._bias              = treelut_model.classes_bias
        self._min, self._max    = treelut_model.min, treelut_model.max


        self._code_gen          = False  # Flag to indicate if code has been generated
        self._prune             = None  # Placeholder for prune vector, to be set later
        self._trees_array       = None  # Placeholder for the trees converted to NumPy arrays
        self._tree_root_indices = None  # Placeholder for root indices of trees




    #===================#
    # Properties       #
    #===================#
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value: TreeLUTClassifier):
        if not isinstance(value, TreeLUTClassifier):
            raise TypeError("Expected a TreeLUTClassifier instance.")
        self._model = value
    

    #===================#
    # Public Methods    #
    #===================#

    def generate_cuda_code(self, output_file: str, optimization: bool = True):
        
        """Generate the CUDA code for the XGBoost model and save it to a file.
        Args:
            output_file (str): The path to the output file where the CUDA code will be saved.
        """
        if not output_file.endswith('.cu'):
            raise ValueError("Output file must have a .cu extension.")
        code = self._header_code()
        code += self._kernel_code() if not optimization else self._optimized_kernel_code()
        code += self._main_code() if not optimization else self._optimized_main_code()

    

        self._convert_trees_to_numpy()  # Convert trees to NumPy arrays for efficient processing
        self._code_gen = True

        with open(output_file, 'w') as f:
            f.write(code)
        print(f"CUDA code generated and saved to {output_file}")

    def calculate_forest_probabilities(self, percentage_to_cut, strategy="adaptive",
                                       level_importance=0.5, progress_importance=0.3,
                                       level_bias=2.0, max_cut_percentage=0.3,
                                       urgency_override_threshold=1.5, output_file='prune.csv'):
        """
        Calculate cut probabilities for all nodes in a forest by walking through them

        General mathematical formula:
        P(cut) = min(1.0, P_base × L_factor × U_factor)
        
        Where:
        - P_base = (total_to_cut - cut_so_far) / (total_nodes - cut_so_far)
        - L_factor = level function (strategy-dependent)
        - U_factor = urgency factor based on progress
        
        Args:
            percentage_to_cut (float): Percentage of nodes to cut from the forest.
            strategy (str): Strategy for calculating cut probabilities ("linear", "exponential", "adaptive", "sigmoid").
            level_importance (float): Controls the importance of the node level in the calculation (0.0 to 1.0+).
            progress_importance (float): Controls the importance of progress in the calculation (0.0 to 1.0+).
            level_bias (float): Base multiplier to give more weight to the node level (1.0+).
            max_cut_percentage (float): Maximum percentage of nodes to cut in each tree (default 0.3).
            urgency_override_threshold (float): Threshold to override urgency factor (default 1.5).
            output_file (str): Path to the output CSV file.

        Returns:
            list of probabilities corresponding to each node
        """
        total_nodes_in_forest = sum(self._model.nodes())
        total_nodes_to_cut = int(total_nodes_in_forest * percentage_to_cut)

        # 2025-07-01 13:42:58
        # Remove leafs and roots from the total nodes to cut
        total_nodes_in_forest -= sum(1 for tree in self._trees for node in tree.values() if node['type'] == 'leaf')
        total_nodes_in_forest -= sum(1 for tree in self._trees for node in tree.keys() if node == 0)  # Remove roots


        nodes_cut_so_far = 0
        nodes_processed_so_far = 0

        probabilities = [0 for i in range(sum(self._model.nodes()))]

        # Initialize hash function for consistent decisions
        pruner = TreePruningHash()

        # 2025-07-18 09:28:08
        # Start from the last tree, that has the less importance
        # zip(range(len(self._trees),-1, -1), reversed(self._trees))
        for tree_index, tree in enumerate(self._trees):
            nodes_in_tree = len(tree)
            nodes_cut_in_tree = 0
            node_levels = self._get_node_levels(tree)
            max_level   = max(node_levels.values())
            # 2025-07-18 09:29:43
            # Start with the last node in a DFS structure 
            for node_id , node in reversed(tree.items()):

                # 2025-06-17 17:22:13
                # If node is a leaf, the probability is 1.0 (no cut)
                if node['type'] == 'leaf':
                    probabilities[node['global_id']] = 0.0
                    continue
                
                # If the node is the root, we do not cut it
                if node_id == 0:
                    probabilities[node['global_id']] = 0.0
                    continue

                # Calculate probability for current node
                if self._debug:
                    print(f"Calculating cut probability for node {node_id} at level {node_levels[node_id]} in tree {tree_index} with global ID {node['global_id']}")

                prob = self._calculate_cut_probability(
                    node_level=node_levels[node_id],
                    nodes_cut_so_far=nodes_cut_so_far,
                    total_nodes_to_cut=total_nodes_to_cut,
                    nodes_processed_so_far=nodes_processed_so_far,
                    total_nodes_in_forest=total_nodes_in_forest,
                    nodes_cut_in_tree=nodes_cut_in_tree,
                    nodes_in_tree=nodes_in_tree,
                    max_cut_percentage=max_cut_percentage,
                    urgency_override_threshold=urgency_override_threshold,
                    strategy=strategy,
                    level_importance=level_importance,
                    progress_importance=progress_importance,
                    level_bias=level_bias,
                    max_level=max_level
                    
                )

                global_id = node['global_id']
                probabilities[global_id] = prob
            
                # 2025-06-16 14:26:34
                # Simulate cutting decision with 10000 random threads id
                decision = self._simulate_cut_decision(pruner, prob, global_id)
                if decision:
                    # Node should be cut
                    nodes_cut_so_far +=  self._consequence_nodes(tree, node_id)
                    nodes_cut_in_tree += self._consequence_nodes(tree, node_id)
            
                nodes_processed_so_far += 1  # Update processed nodes

        # Save probabilities in a .csv file
        with open(output_file, 'w') as f:
            for node_id, prob in enumerate(probabilities):
                f.write(f"{prob}\n")

        self._prune = np.array(probabilities)  # Store probabilities for later use

    def should_cut_node(self, thread_id: int, node_id: int) -> int:
        """Determine if a node should be cut based on its ID and a cut probability.
        Args:
            node_id (int): The unique identifier for the node.

        Returns:
            int: 0 if the node should be cut, 1 if it should not.
        """
        if self._prune is None:
            raise ValueError("Prune vector is not set.")
        cut_probability = self._prune[node_id] if node_id < len(self._prune) else 0.0
        pruner = TreePruningHash()
        return pruner.should_cut_node(thread_id, cut_probability, node_id)

    def should_cut_nodes(self, thread_id: int) -> np.ndarray:
        """
        Determine if a set of nodes should be cut. This vectorized version
        is used by the predict method for high performance.

        Args:
            thread_id (int): The unique identifier for the thread.

        Returns:
            np.ndarray: An array of integers (1 for cut, 0 for keep).
        """
        if not hasattr(self, '_prune') or self._prune is None:
            raise ValueError("Prune vector is not set.")
        if not hasattr(self, '_trees_array') or self._trees_array is None:
            raise RuntimeError("Tree data structure not prepared. Please run generate_cuda_code() after training.")

        GID_IDX = 5

        cut_probabilities = self._prune
        global_ids = self._trees_array[:, GID_IDX].astype(np.int32)
        pruner = TreePruningHash()

        return pruner.should_cut_nodes_vectorized(thread_id, cut_probabilities, global_ids)

    def prune_matrix(self, num_threads: int = None, save_matrix: bool = False) -> np.ndarray:
        """
        Generates a bitmask matrix for the pruning decisions on all nodes in the forest.

        Args:
            num_threads (int): The number of threads to use for the pruning simulation.
            save_matrix (bool): If true, saves the resulting matrix to a CSV file.

        Returns:
            np.ndarray: A 2D array where each row corresponds to a thread and each column is a
                        uint32 integer that packs 32 node pruning decisions as bits.
        """
        if self._prune is None:
            raise ValueError("Prune vector is not set. Please calculate cut probabilities first.")
        if self._trees_array is None:
            raise RuntimeError("Tree data structure not prepared. Please run generate_cuda_code() after training.")
        
        if num_threads is None:
            num_threads = self._n_threads * self._n_blocks  # Default to total threads available

        num_nodes = len(self._prune)
        bits_per_int = np.iinfo(np.uint32).bits

        # Calculate the number of uint32 columns needed to cover all nodes
        num_packed_cols = math.ceil(num_nodes / bits_per_int)

        prune_matrix = np.zeros(shape=(num_threads, num_packed_cols), dtype=np.uint32)


        for thread_id in range(num_threads):
            # This function call remains the same
            should_cut_arr = self.should_cut_nodes(thread_id)

            # 1. Find the indices of the nodes to be cut (where the value is True)
            cut_indices = np.where(should_cut_arr)[0]

            # Only proceed if there are nodes to cut
            if cut_indices.size > 0:
                # 2. Calculate the column index in the packed matrix for each cut
                packed_col_indices = cut_indices // bits_per_int

                # 3. Calculate the bit position (0 to 31) for each cut
                bit_positions = cut_indices % bits_per_int

                # 4. Create the bitmasks (e.g., 1 << 0, 1 << 5, etc.)
                bitmasks = np.left_shift(np.uint32(1), bit_positions)

                # 5. Use np.bitwise_or.at to apply the masks efficiently.
                np.bitwise_or.at(prune_matrix[thread_id], packed_col_indices, bitmasks)
        if save_matrix:
            if not os.path.exists('results'):
                os.makedirs('results')
            
            np.savetxt('results/prune_matrix.csv', prune_matrix, delimiter=',', fmt='%d')


        return prune_matrix
    
    def predict(self, X: np.ndarray, thread_id: int = 0) -> np.ndarray:
        """
        Predict the class labels for the input data using the pruned trees.
        This implementation is vectorized with NumPy and calls the vectorized
        pruning function for high performance.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            thread_id (int): The thread ID to use for pruning decisions.

        Returns:
            np.ndarray: Predicted class labels.
        """
        # Ensure the prune vector and tree arrays are available.
        if not hasattr(self, '_prune') or self._prune is None:
            raise ValueError("Prune vector is not set. Please calculate cut probabilities first.")
        if not hasattr(self, '_trees_array') or self._trees_array is None:
            raise RuntimeError("Tree data structure not prepared. Please run _convert_trees_to_numpy() after training.")

        n_samples = X.shape[0]
        X_quantized = self._quantize_node(X)
        
        # Initialize predictions with class bias
        predictions_agg = np.zeros((n_samples, self._n_classes), dtype=np.float64)
        
        # Initialize with class bias if available
        if hasattr(self, '_bias') and self._bias is not None:
            for class_idx in range(self._n_classes):
                predictions_agg[:, class_idx] = self._bias[class_idx]
        
        # --- Constants for property indices ---
        FEAT_IDX, THRESH_IDX, LEFT_IDX, RIGHT_IDX, VAL_IDX, GID_IDX, IS_LEAF_IDX = 0, 1, 2, 3, 4, 5, 6
                
        # Call the fully vectorized pruning function for maximum performance.
        # # For debugging, assume all prune_flags are False
        # prune_flags = np.zeros(len(global_ids), dtype=bool)  # All False for testing
        # Uncomment this line when ready to use actual pruning:
        prune_flags = self.should_cut_nodes(thread_id).astype(bool)

        # The main prediction loop iterates over each tree in the ensemble.
        for i, root_node_idx in enumerate(self._tree_root_indices):
            class_idx = i % self._n_classes
            
            # Verificação de bounds para segurança
            if root_node_idx >= len(self._trees_array):
                print(f"Warning: Root node {root_node_idx} out of bounds for tree {i}")
                continue
            
            node_indices = np.full(n_samples, root_node_idx, dtype=np.int32)
            active_mask = np.ones(n_samples, dtype=bool)

            iteration = 0
            while np.any(active_mask) and iteration < 100:  # Safety limit
                iteration += 1
                active_indices = node_indices[active_mask]
                
                # Verificação de bounds
                if np.any(active_indices >= len(self._trees_array)) or np.any(active_indices < 0):
                    print(f"Error: Invalid node indices in tree {i}, iteration {iteration}")
                    break
                
                nodes = self._trees_array[active_indices]

                global_ids = nodes[:, GID_IDX].astype(np.int32)
                leaf_flags = nodes[:, IS_LEAF_IDX] == 1
                
                # Since prune_flags is all False, is_finished = leaf_flags
                is_finished = prune_flags[global_ids] | leaf_flags
                
                if np.any(is_finished):
                    finished_indices = active_indices[is_finished]
                    finished_values = self._trees_array[finished_indices, VAL_IDX]
                    
                    # Correct indexing: get the actual sample indices that are finished
                    active_sample_indices = np.where(active_mask)[0]
                    finished_sample_indices = active_sample_indices[is_finished]
                    
                    # Add the leaf values to the predictions
                    predictions_agg[finished_sample_indices, class_idx] += finished_values
                    
                    # Update active mask
                    active_mask[active_mask] = ~is_finished
                    
                    if not np.any(active_mask):
                        break

                # Continue navigation for non-leaf nodes
                if np.any(active_mask):
                    continuing_nodes = self._trees_array[node_indices[active_mask]]
                    
                    features = continuing_nodes[:, FEAT_IDX].astype(np.int32)
                    thresholds = continuing_nodes[:, THRESH_IDX]
                    
                    # Verificação de bounds para features
                    if np.any(features >= X_quantized.shape[1]) or np.any(features < 0):
                        print(f"Error: Invalid feature indices in tree {i}")
                        break
                    
                    # Get feature values for active samples
                    active_sample_indices = np.where(active_mask)[0]
                    sample_features = X_quantized[active_sample_indices, features]
                    left_path_mask = sample_features < thresholds
                    
                    next_nodes = np.where(
                        left_path_mask,
                        continuing_nodes[:, LEFT_IDX],
                        continuing_nodes[:, RIGHT_IDX]
                    ).astype(np.int32)
                    
                    node_indices[active_mask] = next_nodes
            
            if iteration >= 100:
                print(f"Warning: Tree {i} hit iteration limit!")
        
        # Debug info
        # print(f"Final predictions_agg stats per class:")
        # for c in range(self._n_classes):
        #     print(f"  Class {c}: min={np.min(predictions_agg[:, c]):.3f}, "
        #         f"max={np.max(predictions_agg[:, c]):.3f}, "
        #         f"mean={np.mean(predictions_agg[:, c]):.3f}")
        
        return np.argmax(predictions_agg, axis=1)

    #====================#
    # Private Methods   #
    #====================#
    def _convert_trees_to_numpy(self):
        """
        Converts the list of dictionary-based trees into flat NumPy arrays
        for efficient processing. This is a one-time operation that should be
        called after training is complete.
        """
        if not hasattr(self, '_trees') or not self._trees:
            self._trees_array = None
            self._tree_root_indices = None
            return

        node_list = []
        tree_root_indices = []
        
        # --- Constants for property indices ---
        FEAT_IDX, THRESH_IDX, LEFT_IDX, RIGHT_IDX, VAL_IDX, GID_IDX, IS_LEAF_IDX = 0, 1, 2, 3, 4, 5, 6
        
        current_node_offset = 0
        for tree in self._trees:
            tree_root_indices.append(current_node_offset)
            
            node_id_map = {node_id: i + current_node_offset for i, node_id in enumerate(tree.keys())}
            
            for node_id, node in tree.items():
                if node['type'] == 'leaf':
                    node_data = [-1, -1, -1, -1, node['value'], node['global_id'], 1]
                else:
                    node_data = [
                        node['feature'], node['threshold'],
                        node_id_map[node['yes']], node_id_map[node['no']],
                        node['value'], node['global_id'], 0
                    ]
                node_list.append(node_data)
            
            current_node_offset += len(tree)

        self._trees_array = np.array(node_list, dtype=np.float64)
        self._tree_root_indices = np.array(tree_root_indices, dtype=np.int32)

        # Save as a csv file for debugging purposes
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savetxt('results/trees_array.csv', self._trees_array, delimiter=',', fmt='%.6f')

    def _quantize_node(self, X) -> int:
        """Quantize based on min_max scaling the values of each feature and
        but it under the w_feature max bits.
        """

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        X_train_min_max = np.round(scaler.fit_transform(X) * (2**self._w_feature - 1))

        return X_train_min_max

    def _quantize_leafs(self, tree) -> None:
        """
        Quantize using min_max scaling the values of each node and but it under the
        w_tree max bits.
        """
        # Verify if the tree node type split has a value
        for node in tree.values():
            if node['type'] == 'split' and 'value' not in node:
                raise ValueError(f"Node {node['id']} is a split but does not have a value.")
        
        # Get min and max values of the tree
        
        min_value = min(node['value'] for node in tree.values() if node['type'] == 'split')
        max_value = max(node['value'] for node in tree.values() if node['type'] == 'split')

        # Calculate the range and step size for quantization
        value_range = max_value - min_value
        step_size = value_range / (2 ** self._w_tree)
        if value_range == 0:
            step_size = 1

        # Quantize the values of each node
        for node in tree.values():
            if node['type'] == 'split':
                # Quantize the value based on the step size
                node['value'] = int((node['value'] - min_value) / step_size)
                # Ensure the value is within the range of w_tree bits
                node['value'] = max(0, min(node['value'], (1 << self._w_tree) - 1))

    def _get_mean(self, tree: dict, node_id: int) -> int:
        """Get the mean value of a parent node based on its children values.
        If the node's children are not leaves, it will recursively calculate the mean.
        Args:
            tree (dict): The tree structure to calculate the mean for.
        Returns:
            int: The mean value of the node.
        """

        
        if tree[node_id]['type'] == 'leaf':
            return tree[node_id]['value']

        else:
            no_value = self._get_mean(tree, tree[node_id]['no'])
            yes_value = self._get_mean(tree, tree[node_id]['yes'])
            tree[node_id]['value'] = (no_value + yes_value) / 2
            # Ensure the value is an integer with max w_tree bits
            return tree[node_id]['value']
        
    def _calculate_cut_probability(self, node_level, nodes_cut_so_far, total_nodes_to_cut,
                                   nodes_processed_so_far, total_nodes_in_forest,
                                   nodes_cut_in_tree, nodes_in_tree, max_cut_percentage,
                                   urgency_override_threshold, strategy,
                                   level_importance, progress_importance, level_bias,
                                   max_level):
        """
        Calculate cut probability based on tree parameters
        
        General mathematical formula:
        P(cut) = min(1.0, P_base × L_factor × U_factor)
        
        Where:
        - P_base = (total_to_cut - cut_so_far) / (total_nodes - cut_so_far)
        - L_factor = level function (strategy-dependent)
        - U_factor = urgency factor based on progress

        Args:
            node_level: node level in tree (0 = root)
            nodes_cut_so_far: how many nodes have been cut already
            total_nodes_to_cut: target total nodes to cut
            node_cut_in_tree: how many nodes have been cut in the current tree
            max_cut_percentage: maximum percentage of nodes to cut in each tree (default 0.3)
            strategy: calculation strategy ("linear", "exponential", "adaptive", "sigmoid")
            level_importance: controls level impact (0.0 to 1.0+)
            progress_importance: controls progress impact (0.0 to 1.0+)  
            level_bias: base multiplier to give more weight to level (1.0+)
            max_level: maximum level of the tree

        Returns:
            probability between 0.0 and 1.0, beeing 1.0 the node will be cut and 0.0 the node will not be cut.
        """
        # --- Soft Per-Tree Limit Logic ---
        max_nodes_in_tree = int(nodes_in_tree * max_cut_percentage)
        is_tree_limit_reached = nodes_cut_in_tree >= max_nodes_in_tree

        remaining_to_process = total_nodes_in_forest - nodes_processed_so_far
        remaining_to_cut = total_nodes_to_cut - nodes_cut_so_far

        urgency_ratio = remaining_to_cut / remaining_to_process

        if self._debug:
            print(f"  Status:")
            print(f"    · Nodes cut so far: {nodes_cut_so_far}")
            print(f"    · Total nodes to cut: {total_nodes_to_cut}")
            print(f"    · Nodes processed so far: {nodes_processed_so_far}")
            print(f"    · Total nodes in forest: {total_nodes_in_forest}")
            print(f"    · Nodes cut in current tree: {nodes_cut_in_tree}")
            print(f"    · Max nodes to cut in tree: {max_nodes_in_tree}")
            print(f"    · Is tree limit reached: {is_tree_limit_reached}")
            print(f"    · Remaining to process: {remaining_to_process}")
            print(f"    · Remaining to cut: {remaining_to_cut}")

        if remaining_to_process > 0:
            is_desperate = urgency_ratio > ((total_nodes_to_cut / total_nodes_in_forest) * urgency_override_threshold)
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
            level_factor_temp = level_bias + (node_level * level_importance)
            level_factor = level_factor_temp if level_factor_temp > 0 else 1  # 2025-07-01 20:32:18::Avoid always zero final_prob
            progress_ratio = nodes_cut_so_far / total_nodes_to_cut if total_nodes_to_cut > 0 else 0
            urgency_factor = (1.0 + (0.5 - progress_ratio)) * (1.0 + progress_importance)
            urgency_factor = max(0.1, urgency_factor)
        elif strategy == "flat": # Cut just nodes below a level
            if self._debug:
                print(f"    · Node Level: {node_level}")
                print(f"    · Max Level: {max_level}")
                print(f"    · Node Zone: {node_level / max_level}")

            node_zone = node_level / max_level # Just cut the 50%ths deepest nodes
            level_factor = level_bias + (node_level * level_importance) if node_zone >= 0.5 else 0
            progress_ratio = nodes_cut_so_far / total_nodes_to_cut if total_nodes_to_cut > 0 else 0
            urgency_factor = (1.0 + (0.5 - progress_ratio)) * (1.0 + progress_importance)
            urgency_factor = max(0.1, urgency_factor)
        elif strategy == "random":
            return random.uniform(0.0, 1.0)  # Random probability for testing
        

        if self._debug:
            print(f"    » Level Factor: {level_factor:.4f}")
            print(f"    » Urgency Factor: {urgency_factor:.4f}")
            print(f"    » Base Probability: {base_prob:.4f}")
            print(f"    » Final Probability: {base_prob * level_factor * urgency_factor:.4f}")
            
            

        final_prob = base_prob * level_factor * urgency_factor
        return min(1.0, max(0.0, final_prob))

    def _timestamp(self) -> str:
        """Get the current timestamp in a human-readable format.
        Returns:
            str: The current timestamp.
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_node_levels(self, tree: dict) -> dict:
        """
        Get the level of each node in a single tree.

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
    
    def _simulate_cut_decision(self, pruner: TreePruningHash, prob: float, global_id: int) -> int:
        """
        [CORRECTED] Simulates the cut decision across 1000 threads and returns the mode.
        
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
        for thread_id in range(10000):
            decision = pruner.should_cut_node(id=thread_id, cut_probability=prob, salt=global_id)
            decisions.append(decision)
        
        # Count votes and return the majority decision
        counts = collections.Counter(decisions)
        return counts.most_common(1)[0][0]

    def _get_threshold(self, X_min, X_max) -> np.ndarray:
        """
        Calculate the threshold for the quantization module based on the minimum and maximum values of the features.
        The threshold is calculated as the (2**w-feature)-1 midpoints between the minimum and maximum values.
        """

        thresholds = np.zeros( 2**self._w_feature - 1)

        min_val = X_min
        max_val = X_max
        if min_val == max_val - 1:
            # If min and max are one scale apart, set all thresholds to -1
            # This indicates that the feature is constant and does not need quantization.
            thresholds[:] = -1
        else: 
            step = int((max_val - min_val) / (2**self._w_feature - 1))
            thresholds[0] = int(min_val + step / 2)
            for j in range(1, 2**self._w_feature - 1):
                thresholds[j] = thresholds[j-1] + step
        # print(f"Info: Thresholds for quantization module: {thresholds}")
        return thresholds
   
    def _consequence_nodes(self, tree: dict, node: int) -> int:
        """
        Calculates the number of descendant nodes that would be removed as a
        consequence of pruning a specific decision node.

        Args:
            tree (dict): The tree data structure, where keys are node IDs.
            node (int): The ID of the decision node to be pruned.

        Returns:
            int: The number of nodes that would be pruned as a consequence.
        """
        # Nodes that will be removed as a consequence
        consequence_nodes = set()
        
        # Check if the node to be cut exists and is a decision node ('split')
        node_data = tree.get(node)
        if not node_data or node_data.get('type') != 'split':
            # If it's a leaf or doesn't exist, no other nodes are affected
            return 0

        # Initialize a queue for Breadth-First Search (BFS) starting from the node's children
        queue = []
        left_child = node_data.get('no')
        right_child = node_data.get('yes')

        if left_child is not None:
            queue.append(left_child)
        if right_child is not None:
            queue.append(right_child)
        
        # Perform the traversal to find all descendants
        while queue:
            current_node_id = queue.pop(0)
            if current_node_id not in consequence_nodes:                
                consequence_nodes.add(current_node_id)
                
                current_node_data = tree.get(current_node_id)
                
                # If the descendant is also a decision node, add its children to the queue
                if current_node_data and current_node_data.get('type') == 'split':
                    grand_left_child = current_node_data.get('no')
                    grand_right_child = current_node_data.get('yes')
                    
                    if grand_left_child is not None:
                        queue.append(grand_left_child)
                    if grand_right_child is not None:
                        queue.append(grand_right_child)

        # Return the total count of nodes in the subtree
        return len(consequence_nodes)
    
    #====================#
    # Code Gen Methods   #
    #====================#

    def _tree_code(self, tree: dict, id: int, number_trees: int) -> str:
        """Generate CUDA code for a single tree using ternary operations (MUX).

        Args:
            tree (dict): The tree structure to generate code for.
            id (int): The identifier for the tree.
        Returns:
            str: The generated CUDA code for the tree.
        
        Note:
            I.e. of tree structure:
            i.e.:{  0: {'type': 'split', 'feature': 32, 'threshold': 1, 'no': 2, 'yes': 1},
                    1: {'type': 'split',
                    'feature': 33,
                    'threshold': 1,
                    'no': 4,
                    'yes': 3,
                    'parent_node': 0,
                    'parent_yesno': 'yes'},
                    3: {'type': 'leaf', 'value': 0, 'parent_node': 1, 'parent_yesno': 'yes'},
                    4: {'type': 'leaf', 'value': 1, 'parent_node': 1, 'parent_yesno': 'no'},
                    2: {'type': 'split',
                    'feature': 1,
                    'threshold': 1,
                    'no': 6,
                    'yes': 5,
                    'parent_node': 0,
                    'parent_yesno': 'no'},
                    5: {'type': 'leaf', 'value': 1, 'parent_node': 2, 'parent_yesno': 'yes'},
                    6: {'type': 'leaf', 'value': 1, 'parent_node': 2, 'parent_yesno': 'no'}},

        """
        
        self._get_mean(tree, 0)  # Calculate means for the tree
        self._quantize_leafs(tree)
        
        # Calcula o offset baseado no id da árvore atual
        nodes_offset = sum(self._model.nodes()[:id]) if id > 0 else 0
        tree_id = id // self._n_classes      
        class_id = id % self._n_classes      


        root_index = class_id * (len(self._trees) // self._n_classes) + tree_id

        code = ""
        for node_id, node in reversed(tree.items()):  # Topological sort of the tree
            global_id = node_id + nodes_offset

            if node_id == 0:  # Save the result
                # Root do not prune
                code += f"      root[{root_index}] = (features[{node['feature']}] < {node['threshold']}) ? node_{node['yes'] + nodes_offset} : node_{node['no'] + nodes_offset};\n"
            elif node['type'] == 'split':
                code += f"      node_{global_id} = prune_hash(idx, prune[{global_id}], {global_id}) ? {node['value']} : ((features[{node['feature']}] < {node['threshold']}) ? node_{node['yes'] + nodes_offset} : node_{node['no'] + nodes_offset})  ;\n"
            elif node['type'] == 'leaf':
                code += f"      node_{global_id} = {node['value']};\n"
        
        return code

    def _forest_code(self, optimization: bool = False) -> str:
        """Generate CUDA code for the entire forest.
        Returns:
            str: The generated CUDA code for the forest.
        """
        
        code = ""
        for i, tree in enumerate(self._trees):
            code += f"// Tree {i}\n"
            code += self._tree_code(tree, i, len(self._trees)) if not optimization else self._optimized_tree_code(tree, i, len(self._trees))
            code += "\n"
        return code
    
    def sum_code(self) -> str:
        """Generate CUDA code for the sum operation.
        Returns:
            str: The generated CUDA code for the sum operation.
        """

        
        code = "__device__ int sum(int* arr, int start, int stop, int bias) {\n"
        code += "    int total = bias;\n" # 2025-06-11 17:06:36 WARNING: Maybe bias can be a small size type
        code += "    for (int i = start; i < stop; i++) {\n"
        code += "        total += arr[i];\n"
        code += "    }\n"
        code += "    return total;\n"
        code += "}\n\n"
        return code

    def _argmax_code(self):
        """Generate CUDA code for the argmax operation. 
        Returns:
            str: The generated CUDA code for the argmax operation.
        """
        
        code = "__device__ int argmax(int* arr, int size) {\n"
        code += "    int max_index = 0;\n"
        code += "    for (int i = 1; i < size; i++) {\n"
        code += "        if (arr[i] > arr[max_index]) {\n"
        code += "            max_index = i;\n"
        code += "        }\n"
        code += "    }\n"
        code += "    return max_index;\n"
        code += "}\n\n"

        return code

    def _quantization_code(self):
        """Generate CUDA code for the quantization operation, based on the number max of bits (w_feature).
        Returns:
            str: The generated CUDA code for the quantization operation.
        """

 
        
        code = "__device__ void quantize(int* arr, int size) {\n"
        for i in range(len(self._min)):
            threshold = self._get_threshold(self._min[i], self._max[i]) # Getting threshold for feature i
            if threshold[0] == -1:
                # If the feature is constant, set it to 0
                code += f"    arr[{i}] = arr[{i}];\n"
                continue
            code += f"    arr[{i}] = (arr[{i}] < {threshold[0]}) ? 0 :\n"
            for j in range(1, len(threshold) - 1):
                code += f"               (arr[{i}] < {threshold[j]}) ? {j} :\n"
            code += f"               (arr[{i}] < {threshold[-1]}) ? {len(threshold) - 1} : {len(threshold)};\n"
        code += "}\n\n"


        return code

    def _get_prune_code(self) -> str:
        code = "__device__ int prune_hash(int id_val, float prob, int prob_index = 0) {\n"
        code += "   if (id_val == 0) return 0;\n"
        code += "   if (prob <= 0.0f) return 0;\n"
        code += "   if (prob >= 1.0f) return 1;\n"
        code += "   uint64_t combined = ((uint64_t)id_val << 32) | (uint32_t)prob_index;\n"
        code += "   uint32_t hash_val = (uint32_t)(combined ^ (combined >> 32));\n"
        code += "   hash_val ^= hash_val >> 16;\n"
        code += "   hash_val *= 0x85ebca6b;\n"
        code += "   hash_val ^= hash_val >> 13;\n"
        code += "   hash_val *= 0xc2b2ae35;\n"
        code += "   hash_val ^= hash_val >> 16;\n"
        code += "   return ((hash_val & 0x7FFFFFFF) < (uint32_t)(prob * 2147483647.0f)) ? 1 : 0;\n"
        code += "}\n\n"
        return code

    def _read_vector_code(self) -> str:
        """Generate code for a function that reads a .csv and allocate 
        a vector of 1d with its data. This function transforms a matrix
        into a vector, where each row is a sample and each column is a feature.
        Returns:
            str: The generated CUDA code to read the vector from CSV.
        """


        
        code = "template<typename T>\n"
        code += "void read_vector_from_csv(T* vector, const char* filename, int rows, int cols) {\n"
        code += "    FILE* file = fopen(filename, \"r\");\n"
        code += "    if (file == NULL) {\n"
        code += "        printf(\"Error opening file: %s\\n\", filename);\n"
        code += "        exit(EXIT_FAILURE);\n"
        code += "    }\n"
        code += "    char line[1024];\n"
        code += "    char* token;\n"
        code += "    int index = 0;\n"
        code += "    while (fgets(line, sizeof(line), file) != NULL) {\n"
        code += "        token = strtok(line, \",\");\n"
        code += "        while (token != NULL) {\n"
        code += "            if constexpr (std::is_same_v<T, int>) {\n"
        code += "                vector[index++] = atoi(token);\n"
        code += "            } else if constexpr (std::is_same_v<T, float>) {\n"
        code += "                vector[index++] = (float)atof(token);\n"
        code += "            } else if constexpr (std::is_same_v<T, double>) {\n"
        code += "                vector[index++] = atof(token);\n"
        code += "            }\n"
        code += "            token = strtok(NULL, \",\");\n"
        code += "        }\n"
        code += "    }\n"
        code += "    fclose(file);\n"
        code += "}\n\n"

        return code
        
    def _kernel_code(self):
        # Generate the kernel code for the CUDA file

        code = "__global__ void xgboost_kernel(int* X, int* Y, int* Y_expected, float* prune, int total_threads,  int sample_size) {\n"

        code += "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        code += "    if (idx >= total_threads) return;\n\n"
        code += f"   __shared__ int features[{len(self._min)}];\n" # Shared memory for features
        code += f"   int root[{len(self._trees)}];\n" # Root node for each tree
        code += f"   int sum_arr[{self._n_classes}];\n\n" # Sum[classes]
        code += f"   int local_count = 0;\n"  # Local count for correct predictions

        nodes_offset = 0
        for tree_id, tree in enumerate(self._trees):
            # Filtra apenas os nós que não são root (node_id != 0)
            non_root_nodes = [nid for nid in tree.keys() if nid != 0]
            
            for i, node_id in enumerate(non_root_nodes):
                node = tree[node_id]
                global_id = node_id + nodes_offset
                
                if i == 0: # First node is the root node, so we don't need to declare it
                    code += f"   int node_{global_id}"
                elif i == len(non_root_nodes) - 1: # Last node is the leaf node, so we don't need to declare it
                    code += f", node_{global_id};\n"  # Last node is a leaf, so we declare it
                else: # Other nodes
                    code += f", node_{global_id}"
            
            # Update offset using self._model.nodes()
            nodes_offset += self._model.nodes()[tree_id]
            
        code += "\n"  # Add a newline for better readability

        code +=  "   for (int i = 0; i < sample_size; i++) {\n"
        code += "        if(threadIdx.x==0) {\n"
        code += f"       for (int j = 0; j < {len(self._min)}; j++) {{\n"
        code += f"            features[j] = X[i * {len(self._min)} + j];\n"
        code += "        }\n"
        code += "        }\n"
        code += "        __syncthreads();\n"  # Synchronize threads
        code += "        int pred = 0;\n"

        code += f"      {self._forest_code()}"  # 2. Forest code generation
        for class_id in range(self._n_classes): # 3. Sum operation for each class
            code += f"        sum_arr[{class_id}] = sum(root, {class_id * (len(self._trees) // self._n_classes)}, {class_id * (len(self._trees) // self._n_classes) + (len(self._trees) // self._n_classes)}, {self._bias[class_id]});\n"
        code += f"        pred = argmax(sum_arr, {self._n_classes});\n" # 4. Argmax operation to get the predicted class
        code += f"        if(pred == Y_expected[i]) {{\n"  # 5. Check if the prediction is correct
        code += "            local_count++;\n"  # Increment the count for correct predictions
        code += "        }\n"
        code += "    }\n"
        code += "    Y[idx] = local_count;\n"
        code += "}\n\n"
        return code

    def _header_code(self):
        # Generate the header code for the CUDA file
        code = f"""
/*********************************************************************************************
      ___           ___                         ___           ___           ___                          
     /__/|         /  /\         _____         /  /\         /  /\         /  /\          ___            
    |  |:|        /  /:/_       /  /::\       /  /::\       /  /::\       /  /:/_        /  /\           
    |  |:|       /  /:/ /\     /  /:/\:\     /  /:/\:\     /  /:/\:\     /  /:/ /\      /  /:/           
  __|__|:|      /  /:/_/::\   /  /:/~/::\   /  /:/  \:\   /  /:/  \:\   /  /:/ /::\    /  /:/            
 /__/::::\____ /__/:/__\/\:\ /__/:/ /:/\:| /__/:/ \__\:\ /__/:/ \__\:\ /__/:/ /:/\:\  /  /::\            
    ~\~~\::::/ \  \:\ /~~/:/ \  \:\/:/~/:/ \  \:\ /  /:/ \  \:\ /  /:/ \  \:\/:/~/:/ /__/:/\:\           
     |~~|:|~~   \  \:\  /:/   \  \::/ /:/   \  \:\  /:/   \  \:\  /:/   \  \::/ /:/  \__\/  \:\          
     |  |:|      \  \:\/:/     \  \:\/:/     \  \:\/:/     \  \:\/:/     \__\/ /:/        \  \:\         
     |  |:|       \  \::/       \  \::/       \  \::/       \  \::/        /__/:/          \__\/         
     |__|/         \__\/         \__\/         \__\/         \__\/         \__\/                         
                  ___                    ___           ___         ___     
      ___        /  /\                  /  /\         /  /\       /__/\    
     /  /\      /  /::\                /  /:/_       /  /::\      \  \:\   
    /  /:/     /  /:/\:\              /  /:/ /\     /  /:/\:\      \  \:\  
   /  /:/     /  /:/  \:\            /  /:/_/::\   /  /:/~/:/  ___  \  \:\ 
  /  /::\    /__/:/ \__\:\          /__/:/__\/\:\ /__/:/ /:/  /__/\  \__\:
 /__/:/\:\   \  \:\ /  /:/          \  \:\ /~~/:/ \  \:\/:/   \  \:\ /  /:/
 \__\/  \:\   \  \:\  /:/            \  \:\  /:/   \  \::/     \  \:\  /:/ 
      \  \:\   \  \:\/:/              \  \:\/:/     \  \:\      \  \:\/:/  
       \__\/    \  \::/                \  \::/       \  \:\      \  \::/   
                 \__\/                  \__\/         \__\/       \__\/    

File generated by XGBoost2GPU in {self._timestamp()}.

*********************************************************************************************/
"""


        code += "\n\n\n#include <cuda_runtime.h>\n"
        code += "#include <device_launch_parameters.h>\n\n"
        code += "#include <stdio.h>\n"
        code += "#include <cuda_runtime.h>\n"
        code += "#include <stdlib.h>\n"
        code += "#include <cstdlib> // For rand()\n"
        code += "#include <random>\n"
        code += "#include <vector>\n"
        code += "#include <algorithm>\n"
        code += "#include <curand_kernel.h>\n"
        code += "#include <string.h>\n"
        code += "#include <type_traits>\n\n"

        code += "// Utility function to check CUDA errors\n"
        code += "#define CHECK_CUDA_ERROR(call) { \\\n"
        code += "    cudaError_t err = call; \\\n"
        code += "    if (err != cudaSuccess) { \\\n"
        code += "        printf(\"CUDA error: %s\\n\", cudaGetErrorString(err)); \\\n"
        code += "        exit(1); \\\n"
        code += "    } \\\n"
        code += "}\n"

        code += "// Utility functions\n"
        code += self._read_vector_code()
        code += self.sum_code()
        code += self._argmax_code()
        code += self._quantization_code()
        code += self._get_prune_code()

        return code

    def _main_code(self):
        # Generate the main function code for the CUDA file

        code = "int main() {\n"

        code += "    // Define the number of threads and blocks\n"
        code += f"    const int threadsPerBlock = {self._n_threads};\n" # 2025-06-16 22:56:40 WARNING: This is a small number of threads per block, normal is 256
        code += f"    const int optimalBlocks = {self._n_blocks};\n"
        code += "    const int totalThreads = optimalBlocks * threadsPerBlock;\n\n"

        code += "    // Define the size of the input data\n"
        code += f"    const int sample_size = {self._n_samples}; // Number of samples\n"
        code += f"    const int prune_size = {sum(self._model.nodes())}; // Size of the prune vector =  Number of nodes in the forest\n\n"


        code += "    // Initialize CUDA and allocate memory for input and output\n"
        code += "    int* X;\n"
        code += "    int* Y;\n"
        code += "    int* Y_expected;\n"
        code += "    float* prune;\n\n"

        code += f"    int* X_host = (int*)malloc(sample_size * sizeof(int) * {len(self._min)}); // Host input data\n"
        code += "    int* Y_host = (int*)malloc(sizeof(int) * totalThreads); // Host output data\n"
        code += "    int* Y_expected_host = (int*)malloc(sample_size * sizeof(int)); // Host expected output data\n"
        code += "    float* prune_host = (float*)malloc(prune_size * sizeof(float)); // Host prune vector\n\n"

        code += f"    read_vector_from_csv<int>(X_host, \"input.csv\", sample_size, {len(self._min)}); // Read input data from CSV\n"
        code += f"    read_vector_from_csv<int>(Y_expected_host, \"expected_output.csv\", sample_size, 1); // Read expected output data from CSV\n"
        code += f"    read_vector_from_csv<float>(prune_host, \"prune.csv\", prune_size, 1); // Read prune vector from CSV\n\n"

        code += "    // Allocate memory on the device\n"
        code += f"    cudaMalloc((void**)&X, sample_size * sizeof(int) * {len(self._min)});\n"
        code += "    cudaMalloc((void**)&Y, totalThreads * sizeof(int));\n"
        code += "    cudaMalloc((void**)&Y_expected, sample_size * sizeof(int));\n"
        code += "    cudaMalloc((void**)&prune, prune_size * sizeof(float));\n\n"



        code += "    // Copy data from host to device\n"
        code += f"    cudaMemcpy(X, X_host, sample_size * sizeof(int) * {len(self._min)}, cudaMemcpyHostToDevice);\n"
        code += "    cudaMemcpy(Y_expected, Y_expected_host, sample_size * sizeof(int), cudaMemcpyHostToDevice);\n"
        code += "    cudaMemcpy(prune, prune_host, prune_size * sizeof(float), cudaMemcpyHostToDevice);\n\n"

        code += "    // Initialize output array on the device\n"
        code += f"    cudaMemset(Y, 0, totalThreads * sizeof(int));\n\n"

        code += "    // Launch the kernel\n"
        code += "    // Get time\n"
        code += "    cudaEvent_t start, stop;\n"
        code += "    cudaEventCreate(&start);\n"
        code += "    cudaEventCreate(&stop);\n"
        code += "    cudaEventRecord(start);\n\n"
        code += "    xgboost_kernel<<<optimalBlocks, threadsPerBlock>>>(X, Y, Y_expected, prune, totalThreads, sample_size);\n"
        code += "    cudaEventRecord(stop);\n"
        code += "    cudaEventSynchronize(stop);\n\n"

        code += "    // Calculate elapsed time\n"
        code += "    float milliseconds = 0;\n"
        code += "    cudaEventElapsedTime(&milliseconds, start, stop);\n"
        code += "    printf(\"Kernel execution time: %.2f ms\\n\", milliseconds);\n\n"


        code += "    // Check for errors\n"
        code += "    cudaError_t error = cudaGetLastError();\n"
        code += "    if (error != cudaSuccess) {\n"
        code += "        printf(\"CUDA error: %s\\n\", cudaGetErrorString(error));\n"
        code += "        exit(EXIT_FAILURE);\n"
        code += "    }\n\n"

        code += "    // Copy the output data back to the host\n"
        code += "    cudaMemcpy(Y_host, Y, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);\n\n"


        code += "    std::vector<std::pair<int, int>> thread_accuracy(totalThreads);\n";
        code += "    for (int i = 0; i < totalThreads; i++) {\n";
        code += "        thread_accuracy[i] = std::make_pair(Y_host[i], i);\n";
        code += "    }\n";
        code += "    std::sort(thread_accuracy.begin(), thread_accuracy.end(), std::greater<std::pair<int, int>>());\n";

        # Save the all accuracy results to a file
        code += "    FILE *file = fopen(\"accuracy_results.txt\", \"w\");\n"
        code += "    if (file == NULL) {\n"
        code += "        printf(\"Error opening file for writing accuracy results\\n\");\n"
        code += "        exit(EXIT_FAILURE);\n"
        code += "    }\n"

        code += "    fprintf(file, \"id acc\\n\");\n"
        code += "    for (int i = 0; i < totalThreads; i++) {\n"
        code += "        int accuracy = thread_accuracy[i].first;\n"
        code += "        int thread_id = thread_accuracy[i].second;\n"
        code += "        fprintf(file, \"%d %.2f\\n\", thread_id, (float)accuracy / sample_size * 100.0f);\n"
        code += "    }\n"
        code += "    fclose(file);\n"

        code += "    // Reset the device\n"
        code += "    cudaDeviceReset();\n"
        code += "    return 0;\n}"
        return code


    #====================#
    # Optimized Code     #
    # Gen Methods        #
    #====================#

    def _optimized_tree_code(self, tree: dict, id: int, number_trees: int) -> str:
        """Generate CUDA code for a single tree using ternary operations (MUX).

        Args:
            tree (dict): The tree structure to generate code for.
            id (int): The identifier for the tree.
        Returns:
            str: The generated CUDA code for the tree.
        
        Note:
            I.e. of tree structure:
            i.e.:{  0: {'type': 'split', 'feature': 32, 'threshold': 1, 'no': 2, 'yes': 1},
                    1: {'type': 'split',
                    'feature': 33,
                    'threshold': 1,
                    'no': 4,
                    'yes': 3,
                    'parent_node': 0,
                    'parent_yesno': 'yes'},
                    3: {'type': 'leaf', 'value': 0, 'parent_node': 1, 'parent_yesno': 'yes'},
                    4: {'type': 'leaf', 'value': 1, 'parent_node': 1, 'parent_yesno': 'no'},
                    2: {'type': 'split',
                    'feature': 1,
                    'threshold': 1,
                    'no': 6,
                    'yes': 5,
                    'parent_node': 0,
                    'parent_yesno': 'no'},
                    5: {'type': 'leaf', 'value': 1, 'parent_node': 2, 'parent_yesno': 'yes'},
                    6: {'type': 'leaf', 'value': 1, 'parent_node': 2, 'parent_yesno': 'no'}},

        """
        
        self._get_mean(tree, 0)  # Calculate means for the tree
        self._quantize_leafs(tree)
        
        # Calcula o offset baseado no id da árvore atual
        nodes_offset = sum(self._model.nodes()[:id]) if id > 0 else 0
        tree_id = id // self._n_classes      
        class_id = id % self._n_classes      


        root_index = class_id * (len(self._trees) // self._n_classes) + tree_id

        code = ""
        for node_id, node in reversed(tree.items()):  # Topological sort of the tree
            global_id = node_id + nodes_offset

            if node_id == 0:  # Save the result
                # Root do not prune
                code += f"      root[{root_index}] = (features[{node['feature']}] < {node['threshold']}) ? node_{node['yes'] + nodes_offset} : node_{node['no'] + nodes_offset};\n"
            elif node['type'] == 'split':
                code += f"      node_{global_id} = (prune[idx * prune_row_width + ({global_id} / 32)] & (1U << ({global_id} % 32))) ? {node['value']} : ((features[{node['feature']}] < {node['threshold']}) ? node_{node['yes'] + nodes_offset} : node_{node['no'] + nodes_offset})  ;\n"
            elif node['type'] == 'leaf':
                code += f"      node_{global_id} = {node['value']};\n"
        
        return code

    def _optimized_kernel_code(self):
        # Generate the kernel code for the CUDA file

        code = "__global__ void xgboost_kernel(int* X, int* Y, int* Y_expected, const int* prune, int total_threads, int prune_row_width, int sample_size) {\n"

        code += "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        code += "    if (idx >= total_threads) return;\n\n"
        code += f"   __shared__ int features[{len(self._min)}];\n" # Shared memory for features
        code += f"   int root[{len(self._trees)}];\n" # Root node for each tree
        code += f"   int sum_arr[{self._n_classes}];\n\n" # Sum[classes]
        code += f"   int local_count = 0;\n\n"  # Local count for correct predictions

        nodes_offset = 0
        for tree_id, tree in enumerate(self._trees):
            # Filtra apenas os nós que não são root (node_id != 0)
            non_root_nodes = [nid for nid in tree.keys() if nid != 0]
            
            for i, node_id in enumerate(non_root_nodes):
                node = tree[node_id]
                global_id = node_id + nodes_offset
                
                if i == 0: # First node is the root node, so we don't need to declare it
                    code += f"   int node_{global_id}"
                elif i == len(non_root_nodes) - 1: # Last node is the leaf node, so we don't need to declare it
                    code += f", node_{global_id};\n"  # Last node is a leaf, so we declare it
                else: # Other nodes
                    code += f", node_{global_id}"
            
            # Update offset using self._model.nodes()
            nodes_offset += self._model.nodes()[tree_id]
            
        code += "\n"  # Add a newline for better readability

        code +=  "   for (int i = 0; i < sample_size; i++) {\n"
        code += "        if(threadIdx.x==0) {\n"
        code += f"       for (int j = 0; j < {len(self._min)}; j++) {{\n"
        code += f"            features[j] = X[i * {len(self._min)} + j];\n"
        code += "        }\n"
        code += "        }\n"
        code += "        __syncthreads();\n"  # Synchronize threads
        code += "        int pred = 0;\n"

        code += f"      {self._forest_code(optimization=True)}"  # 2. Forest code generation
        for class_id in range(self._n_classes): # 3. Sum operation for each class
            code += f"        sum_arr[{class_id}] = sum(root, {class_id * (len(self._trees) // self._n_classes)}, {class_id * (len(self._trees) // self._n_classes) + (len(self._trees) // self._n_classes)}, {self._bias[class_id]});\n"
        code += f"        pred = argmax(sum_arr, {self._n_classes});\n" # 4. Argmax operation to get the predicted class
        code += f"        if(pred == Y_expected[i]) {{\n"  # 5. Check if the prediction is correct
        code += "            local_count++;\n"  # Increment the count for correct predictions
        code += "        }\n"
        code += "    }\n"
        code += "    Y[idx] = local_count;\n"
        code += "}\n\n"
        return code

    def _optimized_main_code(self):
        # Generate the main function code for the CUDA file

        num_nodes = sum(self._model.nodes())
        bits_per_int = 32  # Number of bits in an integer

        code = "int main() {\n"

        code += "    // Define the number of threads and blocks\n"
        code += f"    const int threadsPerBlock = {self._n_threads};\n" # 2025-06-16 22:56:40 WARNING: This is a small number of threads per block, normal is 256
        code += f"    const int optimalBlocks = {self._n_blocks};\n"
        code += "    const int totalThreads = optimalBlocks * threadsPerBlock;\n\n"

        code += "    // Define the size of the input data\n"
        code += f"    const int sample_size = {self._n_samples}; // Number of samples\n"
        

        code += "    // Initialize CUDA and allocate memory for input and output\n"
        code += "    int* X;\n"
        code += "    int* Y;\n"
        code += "    int* Y_expected;\n"
        code += "    int* prune;\n\n"

        code += f"    int* X_host = (int*)malloc(sample_size * sizeof(int) * {len(self._min)}); // Host input data\n"
        code += "    int* Y_host = (int*)malloc(sizeof(int) * totalThreads); // Host output data\n"
        code += "    int* Y_expected_host = (int*)malloc(sample_size * sizeof(int)); // Host expected output data\n"
        code += f"    int* prune_host = (int*)malloc(totalThreads * {math.ceil(num_nodes / bits_per_int)} * sizeof(int)); // Host prune vector\n\n"

        code += f"    read_vector_from_csv<int>(X_host, \"input.csv\", sample_size, {len(self._min)}); // Read input data from CSV\n"
        code += f"    read_vector_from_csv<int>(Y_expected_host, \"expected_output.csv\", sample_size, 1); // Read expected output data from CSV\n"
        code += f"    read_vector_from_csv<int>(prune_host, \"prune_matrix.csv\", totalThreads, {math.ceil(num_nodes / bits_per_int)}); // Read prune vector from CSV\n\n"

        code += "    \n// Allocate memory on the device\n"
        code += f"    cudaMalloc((void**)&X, sample_size * sizeof(int) * {len(self._min)});\n"
        code += "    cudaMalloc((void**)&Y, totalThreads * sizeof(int));\n"
        code += "    cudaMalloc((void**)&Y_expected, sample_size * sizeof(int));\n"
        code += f"    cudaMalloc((void**)&prune, totalThreads * {math.ceil(num_nodes / bits_per_int)} * sizeof(int));\n\n"



        code += "    // Copy data from host to device\n"
        code += f"    cudaMemcpy(X, X_host, sample_size * sizeof(int) * {len(self._min)}, cudaMemcpyHostToDevice);\n"
        code += "    cudaMemcpy(Y_expected, Y_expected_host, sample_size * sizeof(int), cudaMemcpyHostToDevice);\n"
        code += f"    cudaMemcpy(prune, prune_host, totalThreads * {math.ceil(num_nodes / bits_per_int)} * sizeof(int), cudaMemcpyHostToDevice);\n\n"

        code += "    // Initialize output array on the device\n"
        code += f"    cudaMemset(Y, 0, totalThreads * sizeof(int));\n\n"

        code += "    // Launch the kernel\n"
        code += "    // Get time\n"
        code += "    cudaEvent_t start, stop;\n"
        code += "    cudaEventCreate(&start);\n"
        code += "    cudaEventCreate(&stop);\n"
        code += "    cudaEventRecord(start);\n\n"
        code += f"    xgboost_kernel<<<optimalBlocks, threadsPerBlock>>>(X, Y, Y_expected, prune, totalThreads, {math.ceil(num_nodes / bits_per_int)}, sample_size);\n"
        code += "    cudaEventRecord(stop);\n"
        code += "    cudaEventSynchronize(stop);\n\n"

        code += "    // Calculate elapsed time\n"
        code += "    float milliseconds = 0;\n"
        code += "    cudaEventElapsedTime(&milliseconds, start, stop);\n"
        code += "    printf(\"Kernel execution time: %.2f ms\\n\", milliseconds);\n"

        code += "    // Check for errors\n"
        code += "    cudaError_t error = cudaGetLastError();\n"
        code += "    if (error != cudaSuccess) {\n"
        code += "        printf(\"CUDA error: %s\\n\", cudaGetErrorString(error));\n"
        code += "        exit(EXIT_FAILURE);\n"
        code += "    }\n\n"

        code += "    // Copy the output data back to the host\n"
        code += "    cudaMemcpy(Y_host, Y, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);\n\n"



        # code += "    // Get 5 best accuracy\n";
        code += "    std::vector<std::pair<int, int>> thread_accuracy(totalThreads);\n";
        code += "    for (int i = 0; i < totalThreads; i++) {\n";
        code += "        thread_accuracy[i] = std::make_pair(Y_host[i], i);\n";
        code += "    }\n";
        code += "    std::sort(thread_accuracy.begin(), thread_accuracy.end(), std::greater<std::pair<int, int>>());\n";

        code += "    FILE *file = fopen(\"accuracy_results.txt\", \"w\");\n"
        code += "    if (file == NULL) {\n"
        code += "        printf(\"Error opening file for writing accuracy results\\n\");\n"
        code += "        exit(EXIT_FAILURE);\n"
        code += "    }\n"

        code += "    fprintf(file, \"id acc\\n\");\n"
        code += "    for (int i = 0; i < totalThreads; i++) {\n"
        code += "        int accuracy = thread_accuracy[i].first;\n"
        code += "        int thread_id = thread_accuracy[i].second;\n"
        code += "        fprintf(file, \"%d %.2f\\n\", thread_id, (float)accuracy / sample_size * 100.0f);\n"
        code += "    }\n"
        code += "    fclose(file);\n"

        code += "    // Reset the device\n"
        code += "    cudaDeviceReset();\n"
        code += "    return 0;\n}"
        return code
