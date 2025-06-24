# /*****************************************************************************/
#  * File: treePruningHash.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-06-13
#  * License: [License Type]
#  * Description: Class to determine if a node should be cut based on its ID and a cut probability.
#  *****************************************************************************/

import numpy as np

class TreePruningHash:
    """
    Encapsulates the CUDA-equivalent LCG hashing logic to decide if a node
    or a set of nodes should be pruned based on a probability vector.
    This class is stateless and contains only the computational logic.
    """
    def __init__(self):
        """Initializes the pruner."""
        pass

    #==================#
    # Public Methods   #
    #==================#

    def should_cut_node(self, id: int, cut_probability: float, salt: int = 0) -> int:
        """
        Determine if a single node should be cut based on its ID and cut probability.

        Args:
            node_id (int): The unique identifier for the node.
            cut_probability (float): The probability that this node should be cut.
            salt (int): Optional salt value to add variability.

        Returns:
            int: 1 if the node should be cut, 0 if it should not.
        """
        if id == 0:
            return 0
        if cut_probability >= 1.0:
            return 1
        if cut_probability <= 0.0:
            return 0
        
        hash_value = self._generate_hash_value(id, salt)
        return 1 if (hash_value & 0x7FFFFFFF) < int(cut_probability * 2147483647.0) else 0

    def should_cut_nodes_vectorized(self, id: int, cut_probabilities: np.ndarray, salt: np.ndarray) -> np.ndarray:
        """
        Determine if a set of nodes should be cut. This vectorized version
        is used for high performance predictions.

        Args:
            global_ids (np.ndarray): An array of unique identifiers for the nodes.
            cut_probabilities (np.ndarray): An array of cut probabilities corresponding to global_ids.
            salt (int): Optional salt value to add variability.

        Returns:
            np.ndarray: An array of integers (1 for cut, 0 for keep).
        """
        results = np.zeros_like(cut_probabilities, dtype=int)
        
        if id == 0:
            return results

        # Apply pruning logic
        results[cut_probabilities >= 1.0] = 1
        needs_hashing_mask = (cut_probabilities > 0.0) & (cut_probabilities < 1.0)

        if np.any(needs_hashing_mask):
            salts_to_hash = salt[needs_hashing_mask]
            probs_to_hash = cut_probabilities[needs_hashing_mask]
            
            hash_values = self._generate_hash_value_vectorized(id, salts_to_hash)
            
            thresholds = (probs_to_hash * 2147483647.0).astype(np.uint32)
            decision = (hash_values & 0x7FFFFFFF) < thresholds
            
            results[needs_hashing_mask] = decision.astype(int)
            
        return results # Vector of 1s and 0s indicating which nodes should be cut 

    #==================#
    # Private Methods  #
    #==================#
    
    def _generate_hash_value(self, node_id, salt=0):
        """
        Generate deterministic hash value for the node using ultra-fast LCG.
        Equivalent to the CUDA version.
        """
        # Constants for LCG
        combined = (node_id << 32) | (salt & 0xFFFFFFFF)
        hash_val = (combined ^ (combined >> 32)) & 0xFFFFFFFF
        hash_val ^= hash_val >> 16
        hash_val = (hash_val * 0x85ebca6b) & 0xFFFFFFFF
        hash_val ^= hash_val >> 13
        hash_val = (hash_val * 0xc2b2ae35) & 0xFFFFFFFF
        hash_val ^= hash_val >> 16
        return hash_val
    
    def _generate_hash_value_vectorized(self, id: int, salts=np.array) -> np.ndarray:
        """
        Generate deterministic hash values for an array of nodes using ultra-fast LCG.
        Vectorized version of the CUDA-equivalent hash function.
        """
        shifted_id = np.uint64(id) << 32
        processed_salts = salts.astype(np.uint64) & 0xFFFFFFFF
        combined = shifted_id | processed_salts

        hash_vals = (combined ^ (combined >> 32)).astype(np.uint32)
        hash_vals ^= hash_vals >> 16
        hash_vals = (hash_vals * np.uint32(0x85ebca6b))
        hash_vals ^= hash_vals >> 13
        hash_vals = (hash_vals * np.uint32(0xc2b2ae35))
        hash_vals ^= hash_vals >> 16
        return hash_vals

