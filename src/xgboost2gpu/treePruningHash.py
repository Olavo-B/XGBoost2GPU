# /*****************************************************************************/
#  * File: treePruningHash.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-06-13
#  * License: [License Type]
#  * Description: Class to determine if a node should be cut based on its ID and a cut probability.
#  *****************************************************************************/

class TreePruningHash:
    """
    Class to determine if a node should be cut based on:
    - Node ID
    - Cut probability (calculated externally)

    Avoids creating large matrices using deterministic hash function.
    """

    def __init__(self, seed=42):
        """
        Initialize with a seed for reproducibility
        """
        self.seed = seed

    def should_cut_node(self, id, cut_probability, salt=0):
        """
        Determines if a node should be cut
        Now equivalent to CUDA prune_hash function

        Args:
            node_id: Unique node ID
            cut_probability: cut probability (0.0 to 1.0)
            salt: Optional salt value to add variability (default 0)

        Returns:
            1 if should cut, 0 if should not
        """
        if id == 0:
            return 0
        if cut_probability <= 0:
            return 0
        if cut_probability >= 1:
            return 1

        # Generate deterministic pseudo-random value for this node
        hash_value = self._generate_hash_value(id, salt)

        # Decide based on probability
        return 1 if (hash_value & 0x7FFFFFFF) < int(cut_probability * 2147483647.0) else 0

    def _generate_hash_value(self, node_id, salt=0):
        """
        Generate deterministic hash value for the node using ultra-fast LCG
        Now equivalent to CUDA version
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