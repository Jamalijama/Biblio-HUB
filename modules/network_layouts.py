import math

import networkx as nx
import numpy as np


def _normalize_positions(pos):
    if not pos:
        return {}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    normalized = {}
    for node, (x, y) in pos.items():
        normalized[node] = (
            ((x - min_x) / span_x - 0.5) * 2.0,
            ((y - min_y) / span_y - 0.5) * 2.0,
        )
    return normalized


def _spread_overlaps(pos, min_distance=0.28, iterations=80):
    if not pos:
        return pos
    nodes = list(pos.keys())
    spread = {node: np.array(coords, dtype=float) for node, coords in pos.items()}
    for _ in range(iterations):
        moved = False
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                delta = spread[node_a] - spread[node_b]
                dist = np.linalg.norm(delta)
                if 0 < dist < min_distance:
                    direction = delta / dist
                    shift = (min_distance - dist) * 0.52
                    spread[node_a] = spread[node_a] + direction * shift
                    spread[node_b] = spread[node_b] - direction * shift
                    moved = True
                elif dist == 0:
                    jitter = np.array([0.015, -0.015])
                    spread[node_a] = spread[node_a] + jitter
                    spread[node_b] = spread[node_b] - jitter
                    moved = True
        if not moved:
            break
    return {node: (coords[0], coords[1]) for node, coords in spread.items()}


def _adaptive_spring_iterations(node_count, floor=140, ceiling=320, factor=18):
    node_count = max(int(node_count), 1)
    return max(floor, min(ceiling, int(floor + factor * math.sqrt(node_count))))


def compute_force_layout(G, seed=42, weight_attr="weight"):
    if G.number_of_nodes() == 0:
        return {}
    n = max(G.number_of_nodes(), 1)
    k = 1.75 / math.sqrt(n)
    pos = nx.spring_layout(
        G,
        k=k,
        iterations=_adaptive_spring_iterations(n, floor=160, ceiling=340, factor=18),
        seed=seed,
        weight=weight_attr,
    )
    pos = _normalize_positions(pos)
    return _spread_overlaps(pos, min_distance=0.18)


def _build_cluster_graph(G, group_lookup, weight_attr="weight"):
    meta = nx.Graph()
    for node, cluster_id in group_lookup.items():
        meta.add_node(cluster_id)
    for u, v, data in G.edges(data=True):
        group_u = group_lookup.get(u)
        group_v = group_lookup.get(v)
        if group_u is None or group_v is None or group_u == group_v:
            continue
        weight = data.get(weight_attr, data.get("weight", 1.0))
        if meta.has_edge(group_u, group_v):
            meta[group_u][group_v]["weight"] += weight
        else:
            meta.add_edge(group_u, group_v, weight=weight)
    return meta


def _circle_cluster_centers(cluster_sizes):
    ordered = sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0]))
    count = len(ordered)
    base_radius = max(2.2, 1.6 + count * 0.5)
    centers = {}
    for idx, (cluster_id, size) in enumerate(ordered):
        angle = 2 * math.pi * idx / max(count, 1)
        radius = base_radius + math.sqrt(size) * 0.2
        centers[cluster_id] = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=float)
    return centers


def _cluster_radius(node_count):
    return 0.72 + math.sqrt(max(node_count, 1)) * 0.35


def _compact_cluster_centers(cluster_centers, cluster_sizes, min_gap=0.24, iterations=80):
    if not cluster_centers:
        return {}
    centers = {cid: np.array(coords, dtype=float) for cid, coords in cluster_centers.items()}
    cluster_ids = list(centers.keys())
    for cid, coords in centers.items():
        centers[cid] = coords * 0.68
    for _ in range(iterations):
        moved = False
        for i, cid_a in enumerate(cluster_ids):
            for cid_b in cluster_ids[i + 1:]:
                delta = centers[cid_a] - centers[cid_b]
                dist = np.linalg.norm(delta)
                target = _cluster_radius(cluster_sizes.get(cid_a, 1)) + _cluster_radius(cluster_sizes.get(cid_b, 1)) + min_gap
                if dist == 0:
                    delta = np.array([0.02, -0.02])
                    dist = np.linalg.norm(delta)
                if dist < target:
                    direction = delta / dist
                    shift = (target - dist) * 0.5
                    centers[cid_a] = centers[cid_a] + direction * shift
                    centers[cid_b] = centers[cid_b] - direction * shift
                    moved = True
        if not moved:
            break
    return {cid: coords for cid, coords in centers.items()}


def compute_cluster_layout(G, group_lookup, seed=42, weight_attr="weight"):
    if G.number_of_nodes() == 0:
        return {}
    clusters = {}
    for node, cluster_id in group_lookup.items():
        clusters.setdefault(cluster_id, []).append(node)
    cluster_sizes = {cid: len(nodes) for cid, nodes in clusters.items()}
    if len(clusters) <= 1:
        return compute_force_layout(G, seed=seed, weight_attr=weight_attr)

    meta_graph = _build_cluster_graph(G, group_lookup, weight_attr=weight_attr)
    if meta_graph.number_of_nodes() > 1 and meta_graph.number_of_edges() > 0:
        meta_pos = nx.spring_layout(
            meta_graph,
            k=max(0.8, 1.4 / math.sqrt(meta_graph.number_of_nodes())),
            iterations=_adaptive_spring_iterations(meta_graph.number_of_nodes(), floor=120, ceiling=240, factor=12),
            seed=seed,
            weight="weight",
            scale=1.0,
        )
        meta_pos = _normalize_positions(meta_pos)
        base_spacing = 1.2 + 0.16 * math.sqrt(max(len(clusters), 1))
        cluster_centers = {
            cid: np.array(coords, dtype=float) * (base_spacing + math.sqrt(cluster_sizes.get(cid, 1)) * 0.12)
            for cid, coords in meta_pos.items()
        }
    else:
        cluster_centers = _circle_cluster_centers(cluster_sizes)
    cluster_centers = _compact_cluster_centers(cluster_centers, cluster_sizes)

    combined = {}
    for cluster_id, nodes in clusters.items():
        subgraph = G.subgraph(nodes).copy()
        if subgraph.number_of_nodes() == 1:
            local_pos = {nodes[0]: (0.0, 0.0)}
        else:
            local_pos = nx.spring_layout(
                subgraph,
                k=max(0.9, 2.2 / math.sqrt(subgraph.number_of_nodes())),
                iterations=_adaptive_spring_iterations(subgraph.number_of_nodes(), floor=160, ceiling=340, factor=18),
                seed=seed + int(cluster_id),
                weight=weight_attr,
            )
        local_pos = _normalize_positions(local_pos)
        local_pos = _spread_overlaps(local_pos, min_distance=0.2, iterations=45)
        cluster_radius = _cluster_radius(len(nodes))
        center = cluster_centers.get(cluster_id, np.array([0.0, 0.0]))
        for node, coords in local_pos.items():
            combined[node] = (
                center[0] + coords[0] * cluster_radius,
                center[1] + coords[1] * cluster_radius,
            )
    return _spread_overlaps(combined, min_distance=0.2)


def compute_bipartite_layout(G, left_nodes, right_nodes):
    if G.number_of_nodes() == 0:
        return {}
    left_nodes = [node for node in left_nodes if node in G]
    right_nodes = [node for node in right_nodes if node in G]
    if not left_nodes or not right_nodes:
        return compute_force_layout(G)

    def barycenter_order(nodes, reference_positions):
        scored = []
        for node in nodes:
            weights = []
            for neighbor in G.neighbors(node):
                if neighbor in reference_positions:
                    edge_weight = G[node][neighbor].get("weight", 1.0)
                    weights.extend([reference_positions[neighbor]] * max(1, int(edge_weight)))
            if weights:
                score = sum(weights) / len(weights)
            else:
                score = len(scored)
            scored.append((score, -G.degree(node, weight="weight"), str(node), node))
        scored.sort()
        return [node for _, _, _, node in scored]

    right_reference = {node: idx for idx, node in enumerate(sorted(right_nodes, key=lambda n: -G.degree(n, weight="weight")))}
    left_order = barycenter_order(left_nodes, right_reference)
    left_reference = {node: idx for idx, node in enumerate(left_order)}
    right_order = barycenter_order(right_nodes, left_reference)
    left_order = barycenter_order(left_order, {node: idx for idx, node in enumerate(right_order)})

    pos = {}
    left_span = max(len(left_order) - 1, 1)
    right_span = max(len(right_order) - 1, 1)
    for idx, node in enumerate(left_order):
        y = 1.4 - (2.8 * idx / left_span)
        pos[node] = (-2.1, y)
    for idx, node in enumerate(right_order):
        y = 1.4 - (2.8 * idx / right_span)
        pos[node] = (2.1, y)
    return _spread_overlaps(pos, min_distance=0.18, iterations=30)
