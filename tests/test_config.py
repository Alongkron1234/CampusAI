from campusai import config


def test_config_has_expected_defaults():
    assert config.QDRANT_MODE in {"local", "cloud"}
    assert config.EMBED_MODEL
    assert config.OCR_MODEL
    assert config.GEMINI_MODEL


def test_paths_are_under_project_root():
    assert config.RAW_DIR.is_relative_to(config.ROOT)
    assert config.CHUNKS_PATH.is_relative_to(config.ROOT)
