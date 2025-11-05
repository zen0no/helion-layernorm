import helion
import helion.language as hl
import torch


@helion.kernel(autotune_effort="quick", autotune_random_seed=42)
def layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> [torch.Tensor]:
    out = torch.empty_like(x)

    bs, seq_len = x.shape

    for tile_batch in hl.tile(bs):
        acc_cnt = torch.zeros_like(x[tile_batch, 0], dtype=torch.float32)
        acc_mean = torch.zeros_like(acc_cnt)
        acc_var = torch.zeros_like(acc_cnt)

        for tile_seq in hl.tile(seq_len):
            tile = x[tile_batch, tile_seq]
            tile_sz = tile.size(-1)

            tile_sum = torch.sum(tile, dim=-1)
            tile_sum_sq = torch.sum(tile * tile, dim=-1)

            tile_mean = tile_sum / tile_sz
            tile_var = tile_sum_sq - (tile_sum * tile_sum) / tile_sz

            delta = tile_mean - acc_mean
            new_cnt = acc_cnt + tile_sz
            new_mean = acc_mean + delta * (tile_sz / new_cnt)
            new_var = acc_var + tile_var + delta * delta * (acc_cnt * tile_sz / new_cnt)

            acc_mean, acc_var, acc_cnt = new_mean, new_var, new_cnt

        batch_rstd = torch.rsqrt(acc_var / acc_cnt + eps).unsqueeze(1)
        batch_mean = acc_mean.unsqueeze(1)

        for tile_seq in hl.tile(seq_len):
            tile = x[tile_batch, tile_seq]
            tile_gamma = gamma[tile_seq].unsqueeze(0)
            tile_beta = beta[tile_seq].unsqueeze(0)

            tile = (tile - batch_mean) * batch_rstd
            y = tile * tile_gamma + tile_beta

            out[tile_batch, tile_seq] = y.to(x.dtype)

    return out
