import math

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def combine_tokens(template_tokens, search_tokens, mode='direct', return_res=False):
    # [B, HW, C]
    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    elif mode == 'template_central':
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)
    elif mode == 'partition':
        feat_size_s = int(math.sqrt(len_s))
        feat_size_t = int(math.sqrt(len_t))
        window_size = math.ceil(feat_size_t / 2.)
        # pad feature maps to multiples of window size
        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)
        pad_l = pad_b = pad_r = 0
        # pad_r = (window_size - W % window_size) % window_size
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)
        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)
        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # calculate new h and w, which may be useful for SwinT or others
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError

    return merged_feature


def recover_tokens(merged_tokens, len_template_token, len_search_token, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    elif mode == 'template_central':
        central_pivot = len_search_token // 2
        len_remain = len_search_token - central_pivot
        len_half_and_t = central_pivot + len_template_token

        first_half = merged_tokens[:, :central_pivot, :]
        second_half = merged_tokens[:, -len_remain:, :]
        template_tokens = merged_tokens[:, central_pivot:len_half_and_t, :]

        recovered_tokens = torch.cat((template_tokens, first_half, second_half), dim=1)
    elif mode == 'partition':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError

    return recovered_tokens


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

'''
add token transfer to feature
'''
def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x


'''
feature2token
'''
def feature2token(x):
    if isinstance(x, tuple):
        x = x[0]  # Keep only the tensor part.
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens

# ------------------------------------------------------------------------------------

def build_grid_edge_index(H, W, device):
    """
    Create a 4-neighbor grid graph over an H x W lattice for GAT.
    Return edge_index with shape [2, E].
    """
    edge_index = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i > 0:
                edge_index.append([idx, (i - 1) * W + j])  # Up.
            if i < H - 1:
                edge_index.append([idx, (i + 1) * W + j])  # Down.
            if j > 0:
                edge_index.append([idx, i * W + (j - 1)])  # Left.
            if j < W - 1:
                edge_index.append([idx, i * W + (j + 1)])  # Right.
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(device)
    return edge_index

def build_bimodal_edge_index(H, W, device):
    """
    Build a 2N-node graph where N = H x W and each patch has RGB and TIR modality nodes.
    Node indices [0, N - 1] are RGB and [N, 2N - 1] are TIR.
    The graph contains:
    - 4-neighbor RGB intra-modal edges
    - 4-neighbor TIR intra-modal edges
    - one-to-one RGB <-> TIR cross-modal edges
    """
    def add_edges(offset=0):
        edges = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j + offset
                if i > 0: edges.append([idx, (i - 1) * W + j + offset])
                if i < H - 1: edges.append([idx, (i + 1) * W + j + offset])
                if j > 0: edges.append([idx, i * W + (j - 1) + offset])
                if j < W - 1: edges.append([idx, i * W + (j + 1) + offset])
        return edges

    N = H * W
    edges_rgb = add_edges(offset=0)
    edges_dte = add_edges(offset=N)
    edges_cross = [[i, i + N] for i in range(N)] + [[i + N, i] for i in range(N)]

    all_edges = edges_rgb + edges_dte + edges_cross
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().to(device)  # [2, E]
    return edge_index


def build_bimodal_edge_index_with_semantic(H, W, feat_cat, device, topk=6):
    """
    Build a multimodal graph structure for RGB + TIR:
    - fixed spatial adjacency edges with 4-neighbor connectivity
    - cross-modal edges at corresponding positions
    - semantic edges based on top-k feature similarity

    Args:
    - H, W: image height and width
    - feat_cat: fused features with shape [2N, C], one representation per patch
    - device: target device
    - topk: number of nearest feature neighbors used for semantic edges
    """

    N = H * W
    total_nodes = 2 * N
    edge_index = []

    # --- 1. RGB intra-modal edges ---
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i > 0:
                edge_index.append([idx, (i - 1) * W + j])
            if i < H - 1:
                edge_index.append([idx, (i + 1) * W + j])
            if j > 0:
                edge_index.append([idx, i * W + (j - 1)])
            if j < W - 1:
                edge_index.append([idx, i * W + (j + 1)])

    # --- 2. TIR intra-modal edges ---
    for i in range(H):
        for j in range(W):
            idx = N + i * W + j
            if i > 0:
                edge_index.append([idx, N + (i - 1) * W + j])
            if i < H - 1:
                edge_index.append([idx, N + (i + 1) * W + j])
            if j > 0:
                edge_index.append([idx, N + i * W + (j - 1)])
            if j < W - 1:
                edge_index.append([idx, N + i * W + (j + 1)])

    # --- 3. Cross-modal edges between corresponding positions ---
    for i in range(N):
        edge_index.append([i, i + N])  # RGB -> TIR
        edge_index.append([i + N, i])  # TIR -> RGB

    # --- 4. Add semantic edges based on feature cosine similarity ---
    with torch.no_grad():
        feat = F.normalize(feat_cat, dim=1)  # [2N, C]
        sim = torch.matmul(feat, feat.T)  # [2N, 2N]
        _, topk_indices = torch.topk(sim, k=topk + 1, dim=-1)  # Includes self.

        for src in range(total_nodes):
            for dst in topk_indices[src][1:]:  # Remove self.
                edge_index.append([src, dst.item()])

    # --- Build the edge_index tensor ---
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(device)  # [2, E]
    return edge_index

# def modality_judge_masked(
#     rgb_feat, rgb_mask,
#     tir_feat, tir_mask,
#     eps=1e-6, return_all=False
# ):
#     """
#     Estimate modality reliability from decoupled masks; higher scores are more reliable.
#
#     Inputs:
#         rgb_feat: [B, C, H, W]
#         rgb_mask: [B, 1, H, W]
#         tir_feat: [B, C, H, W]
#         tir_mask: [B, 1, H, W]
#
#     Outputs:
#         score_rgb, score_tir: [B], higher means more reliable.
#
#     Optional:
#         return_all=True returns a dictionary with all score components.
#     """
#
#     def compute_score(feat, mask):
#         inside = (feat * mask).pow(2).mean(dim=[1, 2, 3])
#         outside = (feat * (1 - mask)).pow(2).mean(dim=[1, 2, 3])
#         fg_ratio = inside / (outside + eps)
#
#         coverage = mask.mean(dim=[1, 2, 3])
#         coverage_score = - (coverage - 0.15).abs()
#
#         energy = feat.pow(2).mean(dim=1)
#         smooth = F.avg_pool2d(energy.unsqueeze(1), 5, 1, 2).squeeze(1)
#         structure = -(energy - smooth).abs().mean(dim=[1, 2])
#
#         total_score = fg_ratio + 0.5 * coverage_score + 0.5 * structure
#
#         return total_score, fg_ratio, coverage_score, structure
#
#     score_rgb, fg_r_rgb, cov_rgb, stru_rgb = compute_score(rgb_feat, rgb_mask)
#     score_tir, fg_r_tir, cov_tir, stru_tir = compute_score(tir_feat, tir_mask)
#
#     if return_all:
#         return {
#             "score_rgb": score_rgb,
#             "score_tir": score_tir,
#             "rgb_detail": {
#                 "fg_ratio": fg_r_rgb,
#                 "coverage_score": cov_rgb,
#                 "structure_score": stru_rgb,
#             },
#             "tir_detail": {
#                 "fg_ratio": fg_r_tir,
#                 "coverage_score": cov_tir,
#                 "structure_score": stru_tir,
#             }
#         }
#     else:
#         return score_rgb, score_tir

# ------------------------------------------------------------------------------------

def build_multi_relational_edge_index(H, W, feat_cat, device, topk=6):

    N = H * W
    total_nodes = 2 * N

    # --- 1. Vectorized grid-edge construction for spatial adjacency ---
    # Generate the node-index matrix [H, W].
    node_indices = torch.arange(N, device=device).view(H, W)

    # Extract right and bottom neighbors.
    # Horizontal and vertical neighbors.
    right_src = node_indices[:, :-1].reshape(-1)
    right_dst = node_indices[:, 1:].reshape(-1)
    bottom_src = node_indices[:-1, :].reshape(-1)
    bottom_dst = node_indices[1:, :].reshape(-1)

    # # Optional diagonal neighbors for 8-neighborhood connectivity.
    # # Top-left -> bottom-right.
    # tl_br_src = node_indices[:-1, :-1].reshape(-1)
    # tl_br_dst = node_indices[1:, 1:].reshape(-1)
    # # Top-right -> bottom-left.
    # tr_bl_src = node_indices[:-1, 1:].reshape(-1)
    # tr_bl_dst = node_indices[1:, :-1].reshape(-1)


    # Concatenate and make the graph bidirectional.
    # 4-neighbor 2D grid edges.
    spatial_src = torch.cat([right_src, bottom_src])
    spatial_dst = torch.cat([right_dst, bottom_dst])

    # spatial_src = torch.cat([right_src, bottom_src, tl_br_src, tr_bl_src])
    # spatial_dst = torch.cat([right_dst, bottom_dst, tl_br_dst, tr_bl_dst])


    grid_edges_src = torch.cat([spatial_src, spatial_dst])
    grid_edges_dst = torch.cat([spatial_dst, spatial_src])

    # Copy the RGB grid edges to TIR by offsetting indices by N.
    rgb_edges = torch.stack([grid_edges_src, grid_edges_dst], dim=0)
    tir_edges = rgb_edges + N

    # --- 2. Cross-modal edges (RGB <-> TIR) ---
    cross_src = torch.arange(N, device=device) #[0, 1, 2, ..., N-1]
    cross_dst = cross_src + N #[N, N+1, N+2, ..., 2N-1]
    cross_edges_fwd = torch.stack([cross_src, cross_dst], dim=0)
    cross_edges_bwd = torch.stack([cross_dst, cross_src], dim=0)

    # --- 3. Dynamic semantic edges based on current features ---
    with torch.no_grad():
        feat = F.normalize(feat_cat, dim=1)  # [2N, C]
        sim = torch.matmul(feat, feat.T) # Compute all-pairs dot products.
        # Mask self-connections by setting the diagonal to -inf.
        sim.fill_diagonal_(-float('inf'))
        # Select the top-K most similar nodes for sparse semantic edges.
        _, topk_indices = torch.topk(sim, k=topk, dim=-1)

        # Build edge_index in a vectorized form.
        semantic_src = torch.arange(total_nodes, device=device).unsqueeze(1).repeat(1, topk).flatten()
        semantic_dst = topk_indices.flatten()
        semantic_edges = torch.stack([semantic_src, semantic_dst], dim=0)

    # --- Merge all edge types ---
    all_edges = torch.cat([rgb_edges, tir_edges, cross_edges_fwd, cross_edges_bwd, semantic_edges], dim=1)
    return all_edges


def save_layer_feature_map(tokens, lens_z, lens_x, layer_idx, mode_name, save_dir, scale_factor=20):
    # Keep scale_factor=20 so exported maps are large enough.

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    token = tokens[0]
    z_token = token[:lens_z, :]
    x_token = token[lens_z:, :]

    hz = wz = int(lens_z ** 0.5)
    hx = wx = int(lens_x ** 0.5)

    z_map = z_token.mean(dim=-1).reshape(hz, wz).detach().cpu().numpy()
    x_map = x_token.mean(dim=-1).reshape(hx, wx).detach().cpu().numpy()

    def normalize(data):
        min_v, max_v = data.min(), data.max()
        if max_v - min_v < 1e-8:
            return np.zeros_like(data, dtype=np.uint8)
        data = (data - min_v) / (max_v - min_v)
        return (data * 255).astype(np.uint8)

    z_img = normalize(z_map)
    x_img = normalize(x_map)

    # Upscale the image while preserving the blocky patch style.
    z_img_big = cv2.resize(z_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    x_img_big = cv2.resize(x_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    z_color = cv2.applyColorMap(z_img_big, cv2.COLORMAP_JET)
    x_color = cv2.applyColorMap(x_img_big, cv2.COLORMAP_JET)

    # ================= Removed the step suffix from saved filenames. =================
    filename_z = f"layer_{layer_idx}_{mode_name}_z.png"
    filename_x = f"layer_{layer_idx}_{mode_name}_x.png"
    # ========================================================

    cv2.imwrite(f"{save_dir}/{filename_z}", z_color)
    cv2.imwrite(f"{save_dir}/{filename_x}", x_color)