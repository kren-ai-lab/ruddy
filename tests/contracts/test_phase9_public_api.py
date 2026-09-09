from ruddy import AnalysisBlock, UnifiedAnalysisResult, analyze


def test_phase9_public_symbols_are_importable():
    assert AnalysisBlock.PROFILING.value == "profiling"
    assert UnifiedAnalysisResult.__name__ == "UnifiedAnalysisResult"
    assert callable(analyze)
