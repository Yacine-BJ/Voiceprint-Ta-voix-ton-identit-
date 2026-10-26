from __future__ import annotations

import numpy as np
from pathlib import Path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcule la similarité cosinus entre deux vecteurs d'embedding."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verify_audio(test_embedding: np.ndarray, enrolled_path: Path, threshold: float = 0.75) -> tuple[bool, float]:
    # Charger les voiceprints enrôlés
    data = np.load(enrolled_path)

    if "yacine" not in data:
        raise ValueError("Le locuteur 'yacine' n'est pas trouvé dans les enrollements.")

    enrolled_vector = data["yacine"]

    # Calcul du score
    score = cosine_similarity(test_embedding, enrolled_vector)
    is_authenticated = score >= threshold

    return is_authenticated, score


if __name__ == "__main__":
    # Test fictif avec un vecteur aléatoire (à remplacer par l'extraction réelle de votre modèle)
    dummy_test_embedding = np.random.randn(120).astype(np.float32)

    enrolled_file = Path("/home/yacine/data_science/Voiceprint-Ta-voix-ton-identit-/data/processed/enrolled_voiceprints.npz")

    authenticated, score = verify_audio(dummy_test_embedding, enrolled_file)
    print(f"Score de similarité : {score:.4f}")
    print(f"Accès autorisé ? {'OUI 🟢' if authenticated else 'NON 🔴'}")