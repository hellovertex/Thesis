class Parser:
  """Baseclass to parse files storing played hands as crawled from different poker websites."""
  def parse_file(self, file_path):
    """Reads file and returns and iterator over the played hands.

    Args:
      file_path: path to the database file.
    Returns:

    """
    raise NotImplementedError
