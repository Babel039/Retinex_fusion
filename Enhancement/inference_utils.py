import torch


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size, stride))
    starts.append(length - tile_size)
    return starts


def tiled_forward(x, model, tile_size, overlap=32):
    """Run a scale-1 image model on overlapping tiles and average overlaps."""
    if tile_size <= 0:
        return model(x)
    if overlap < 0 or overlap >= tile_size:
        raise ValueError('tile overlap must be non-negative and smaller than tile size')

    height, width = x.shape[-2:]
    if height <= tile_size and width <= tile_size:
        return model(x)

    stride = tile_size - overlap
    height_starts = _tile_starts(height, tile_size, stride)
    width_starts = _tile_starts(width, tile_size, stride)
    output = None
    weights = None

    for top in height_starts:
        bottom = min(top + tile_size, height)
        for left in width_starts:
            right = min(left + tile_size, width)
            restored_tile = model(x[:, :, top:bottom, left:right])
            if restored_tile.shape[-2:] != (bottom - top, right - left):
                raise ValueError('tiled_forward only supports scale-1 models')
            if output is None:
                output = restored_tile.new_zeros(
                    (restored_tile.shape[0], restored_tile.shape[1], height, width))
                weights = restored_tile.new_zeros((1, 1, height, width))
            output[:, :, top:bottom, left:right].add_(restored_tile)
            weights[:, :, top:bottom, left:right].add_(1)

    return output / weights


def self_ensemble(x, model, forward_fn=None):
    """Average eight flip/rotation variants without stacking their outputs."""
    if forward_fn is None:
        forward_fn = lambda value, network: network(value)

    total = None
    count = 0
    for hflip in (False, True):
        for vflip in (False, True):
            for rotate in (False, True):
                transformed = x
                if hflip:
                    transformed = torch.flip(transformed, (-2,))
                if vflip:
                    transformed = torch.flip(transformed, (-1,))
                if rotate:
                    transformed = torch.rot90(transformed, dims=(-2, -1))

                restored = forward_fn(transformed, model)
                if rotate:
                    restored = torch.rot90(restored, dims=(-2, -1), k=3)
                if vflip:
                    restored = torch.flip(restored, (-1,))
                if hflip:
                    restored = torch.flip(restored, (-2,))

                if total is None:
                    total = torch.zeros_like(restored)
                total.add_(restored)
                count += 1

    return total.div_(count)
