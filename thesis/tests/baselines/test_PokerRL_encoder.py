def test_make_ante(multiply_by=100):
    ante1 = "$0.00"
    ante2 = "$0.01"
    assert float(ante1.split("$")[1]) * multiply_by == 0.00
    assert float(ante2.split("$")[1]) * multiply_by == 1.00