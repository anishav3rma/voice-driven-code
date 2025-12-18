from s2c.asr import decode_mock

def test_decode_mock_severity_changes_output():
    u = "add parameter min_count default two to word_count"
    n1 = decode_mock(u, n_best=5, severity=0.0)["nbest"]
    n2 = decode_mock(u, n_best=5, severity=1.0)["nbest"]
    assert n1 != n2
