from solution.raster import iter_window_specs


def test_window_iterator_covers_edges_without_duplicates():
    specs = list(iter_window_specs(width=1000, height=700, tile_size=640, overlap=128))

    assert specs
    assert max(row + height for row, col, height, width in specs) == 700
    assert max(col + width for row, col, height, width in specs) == 1000
    assert all(height <= 640 and width <= 640 for row, col, height, width in specs)

