"""
Automated unit verification for FaceEngine:
- Model loading
- Feature extraction & alignment simulation
- Profile registration & persistence
- Cosine similarity matching
- Profile deletion
"""

import os
import shutil
import numpy as np
from face_engine import FaceEngine


def run_tests():
    print(">>> Starting FaceEngine Verification Test...")

    test_db = os.path.join("data", "test_faces_db.pkl")
    test_crops_dir = os.path.join("data", "test_registered_faces")

    # Clean previous test artifacts
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(test_crops_dir):
        shutil.rmtree(test_crops_dir, ignore_errors=True)

    # 1. Initialize Engine
    engine = FaceEngine(
        db_path=test_db,
        faces_dir=test_crops_dir,
        cosine_threshold=0.38,
    )
    assert engine.detector is not None, "Detector failed to initialize"
    assert engine.recognizer is not None, "Recognizer failed to initialize"
    print("[PASS] Models successfully loaded.")

    # 2. Test Registration with simulated 128-D normalized embeddings
    np.random.seed(42)
    sample_feat_1 = np.random.randn(1, 128).astype(np.float32)
    sample_feat_1 /= np.linalg.norm(sample_feat_1)

    # Slightly perturbed sample for the same person
    sample_feat_2 = sample_feat_1 + np.random.randn(1, 128).astype(np.float32) * 0.05
    sample_feat_2 /= np.linalg.norm(sample_feat_2)

    dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)

    engine.register_person("Alex Mercer", [sample_feat_1, sample_feat_2], [dummy_crop])
    assert "Alex Mercer" in engine.database, "Alex Mercer not registered in database"
    assert os.path.exists(test_db), "Database file was not saved to disk"
    assert os.path.exists(os.path.join(test_crops_dir, "Alex Mercer", "sample_1.jpg")), "Face crop not saved"
    print("[PASS] Person registration and disk persistence passed.")

    # 3. Test Matching (Should match Alex Mercer)
    query_match = sample_feat_1 + np.random.randn(1, 128).astype(np.float32) * 0.02
    query_match /= np.linalg.norm(query_match)

    matched_name, score, is_match = engine.match_face(query_match)
    print(f"Match query result: name={matched_name}, score={score:.4f}, is_match={is_match}")
    assert matched_name == "Alex Mercer", f"Expected Alex Mercer, got {matched_name}"
    assert is_match is True, "Expected positive match"
    print("[PASS] Known face matching passed.")

    # 4. Test Non-match (Orthogonal random feature -> Unknown)
    unrelated_feat = np.random.randn(1, 128).astype(np.float32)
    unrelated_feat /= np.linalg.norm(unrelated_feat)
    name_unrelated, score_unrelated, is_match_unrelated = engine.match_face(unrelated_feat)
    print(f"Unknown query result: name={name_unrelated}, score={score_unrelated:.4f}, is_match={is_match_unrelated}")
    assert name_unrelated == "Unknown", f"Expected Unknown, got {name_unrelated}"
    assert is_match_unrelated is False, "Expected no match"
    print("[PASS] Unknown face rejection passed.")

    # 5. Test Deletion
    del_ret = engine.delete_person("Alex Mercer")
    assert del_ret is True, "Deletion return was False"
    assert "Alex Mercer" not in engine.database, "Alex Mercer still in database"
    assert not os.path.exists(os.path.join(test_crops_dir, "Alex Mercer")), "Photos folder not cleaned up"
    print("[PASS] Person deletion passed.")

    # Clean up test artifacts
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(test_crops_dir):
        shutil.rmtree(test_crops_dir, ignore_errors=True)

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_tests()
