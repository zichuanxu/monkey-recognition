"""Simplified feature database for monkey face recognition."""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging import LoggerMixin


class FeatureDatabase(LoggerMixin):
    """Simplified database for storing and retrieving monkey face features."""

    def __init__(self):
        """Initialize feature database."""
        self.features = defaultdict(list)  # monkey_id -> list of features
        self.feature_metadata = defaultdict(list)  # monkey_id -> list of metadata

    def add_feature(self, monkey_id: str, feature: np.ndarray, metadata: Optional[Dict] = None):
        """Add a feature vector for a monkey.

        Args:
            monkey_id: Monkey identifier.
            feature: Feature vector.
            metadata: Optional metadata.
        """
        self.features[monkey_id].append(feature.copy())
        self.feature_metadata[monkey_id].append(metadata or {})

    def add_monkey(self, monkey_id: str, features: np.ndarray, metadata: Optional[List[Dict]] = None):
        """Add multiple features for a monkey.

        Args:
            monkey_id: Monkey identifier.
            features: Array of feature vectors (N x feature_dim).
            metadata: Optional list of metadata dicts.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        for i, feature in enumerate(features):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            self.add_feature(monkey_id, feature, meta)

    def find_closest_match(
        self,
        query_feature: np.ndarray,
        top_k: int = 1,
        threshold: float = 0.5
    ) -> List[Tuple[str, float, Dict]]:
        """Find closest matching monkey(s) for a query feature.

        Args:
            query_feature: Query feature vector.
            top_k: Number of top matches to return.
            threshold: Similarity threshold.

        Returns:
            List of (monkey_id, similarity, metadata) tuples.
        """
        if not self.features:
            return []

        matches = []
        query_feature = query_feature.reshape(1, -1)

        for monkey_id, feature_list in self.features.items():
            if not feature_list:
                continue

            # Calculate similarities with all features for this monkey
            features_array = np.vstack(feature_list)
            similarities = cosine_similarity(query_feature, features_array)[0]

            # Take the maximum similarity
            max_similarity = np.max(similarities)
            best_idx = np.argmax(similarities)

            if max_similarity >= threshold:
                metadata = self.feature_metadata[monkey_id][best_idx]
                matches.append((monkey_id, max_similarity, metadata))

        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def get_monkey_ids(self) -> List[str]:
        """Get list of all monkey IDs in database."""
        return list(self.features.keys())

    def get_total_features(self) -> int:
        """Get total number of features in database."""
        return sum(len(feature_list) for feature_list in self.features.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_monkeys': len(self.features),
            'total_features': self.get_total_features(),
            'features_per_monkey': {
                monkey_id: len(feature_list)
                for monkey_id, feature_list in self.features.items()
            }
        }

    def save(self, filepath: str):
        """Save database to file.

        Args:
            filepath: Path to save database.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'features': dict(self.features),
            'feature_metadata': dict(self.feature_metadata)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Feature database saved to {filepath}")

    def load(self, filepath: str):
        """Load database from file.

        Args:
            filepath: Path to load database from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Database file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.features = defaultdict(list, data['features'])
        self.feature_metadata = defaultdict(list, data['feature_metadata'])

        self.logger.info(f"Feature database loaded from {filepath}")

    def clear(self):
        """Clear all data from database."""
        self.features.clear()
        self.feature_metadata.clear()

    def remove_monkey(self, monkey_id: str) -> bool:
        """Remove a monkey from database.

        Args:
            monkey_id: Monkey identifier to remove.

        Returns:
            True if monkey was removed, False if not found.
        """
        if monkey_id in self.features:
            del self.features[monkey_id]
            del self.feature_metadata[monkey_id]
            return True
        return False