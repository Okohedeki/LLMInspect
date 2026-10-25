from toml_lite import parse_line


def test_integer():
    assert parse_line("port = 8080") == ("port", 8080)


def test_booleans():
    assert parse_line("debug = true") == ("debug", True)
    assert parse_line("verbose = false") == ("verbose", False)


def test_string():
    assert parse_line('name = "brainscan"') == ("name", "brainscan")


def test_blank_and_comment_lines():
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("# just a comment") is None


def test_trailing_comment():
    assert parse_line("retries = 3  # try a few times") == ("retries", 3)


def test_hash_and_equals_inside_string():
    assert parse_line('url = "http://x.test/?a=1#top"') == ("url", "http://x.test/?a=1#top")
