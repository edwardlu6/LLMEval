from llmreval.metrics import f1_em

def test_f1_em_basic():
    m = f1_em("Barack Obama was born in Hawaii", "Obama born in Hawaii")
    assert 0.0 <= m["f1"] <= 1.0

