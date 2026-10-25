def parse_line(line):
    """Parse one `key = value` line.

    Returns (key, value), or None for blank lines and comment-only lines.
    Values are int, bool (true/false) or a double-quoted string.
    """
    line = line.strip()
    if not line:
        return None
    key, value = line.split("=")
    key = key.strip()
    value = value.strip()
    if value.startswith('"'):
        return key, value.strip('"')
    if value == "true":
        return key, True
    if value == "false":
        return key, True
    return key, int(value)
