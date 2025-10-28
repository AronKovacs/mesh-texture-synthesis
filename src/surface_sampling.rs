use std::collections::HashMap;

use cgmath::prelude::*;
use cgmath::{Point3, Vector3};
use ndarray::ArrayView2;

use crate::utils::*;

pub fn trace_surface_line_with_samples(
    samples: &mut Vec<(i32, Point3<f32>)>,
    mut pos: Point3<f64>,
    mut dir: Vector3<f64>,
    mut length: f64,
    sample_distance: f64,
    mut triangle: i32,
    vertices_world_positions: ArrayView2<f32>,
    indices: ArrayView2<i32>,
    merged_vertices: &HashMap<Point3<u32>, Vec<i32>>,
    triangle_edge_neighborhoods: &[(i32, i8)],
) -> (i32, Point3<f64>, Vector3<f64>, bool) {
    samples.clear();
    samples.push((triangle, pos.map(|n| n as f32)));

    let mut success;

    loop {
        let segment_length = if length > sample_distance {
            length -= sample_distance;
            sample_distance
        } else {
            let segment_length = length;
            length = 0.0;
            segment_length
        };

        let (new_triangle, new_pos, new_dir, new_success) = trace_surface_line(
            pos,
            dir,
            segment_length,
            triangle,
            vertices_world_positions,
            indices,
            merged_vertices,
            triangle_edge_neighborhoods,
        );

        success = new_success;
        if !success {
            break;
        }

        samples.push((new_triangle, new_pos.map(|n| n as f32)));

        triangle = new_triangle;
        pos = new_pos;
        dir = new_dir;

        if length <= 0.0 {
            break;
        }
    }

    (triangle, pos, dir, success)
}

pub fn trace_surface_line(
    mut pos: Point3<f64>,
    mut dir: Vector3<f64>,
    mut length: f64,
    mut triangle: i32,
    vertices_world_positions: ArrayView2<f32>,
    indices: ArrayView2<i32>,
    merged_vertices: &HashMap<Point3<u32>, Vec<i32>>,
    triangle_edge_neighborhoods: &[(i32, i8)],
) -> (i32, Point3<f64>, Vector3<f64>, bool) {
    let mut successful = true;
    let eps = 0.0001;

    let loop_counter_max = 100000;

    'primary: for loop_counter in 0..loop_counter_max {
        if loop_counter == loop_counter_max - 1 {
            successful = false;
            break 'primary;
        }

        if length <= 0.0 {
            break;
        }

        dir = dir.normalize();

        // get world positions of the current triangle
        let vertices_wp: [Point3<f32>; 3] = std::array::from_fn(|i| {
            Point3::new(
                vertices_world_positions[[indices[[triangle as usize, i]] as usize, 0]],
                vertices_world_positions[[indices[[triangle as usize, i]] as usize, 1]],
                vertices_world_positions[[indices[[triangle as usize, i]] as usize, 2]],
            )
        });

        // compute normal of the current triangle
        let normal_in = (vertices_wp[1] - vertices_wp[0])
            .cross(vertices_wp[2] - vertices_wp[0])
            .map(|n| n as f64)
            .normalize();

        // check if the current pos is directly on a vertex
        let is_on_vertex = {
            let mut is_on_vertex = None;
            let mut min_distance = std::f64::INFINITY;
            for i in [0, 1, 2] {
                let distance = pos.distance(vertices_wp[i].map(|n| n as f64));
                if distance < min_distance && distance < eps {
                    min_distance = distance;
                    is_on_vertex = Some(i);
                }
            }
            is_on_vertex
        };

        // check if the current pos is directly on an edge
        let is_on_edge = {
            let mut is_on_edge = None;
            let mut min_distance = std::f64::INFINITY;
            for ((edge_start, edge_end), opposite_vertex) in [((0, 1), 2), ((1, 2), 0), ((2, 0), 1)]
            {
                let plane_normal = oriented_plane_normal(
                    vertices_wp[opposite_vertex].map(|n| n as f64),
                    vertices_wp[edge_start].map(|n| n as f64),
                    vertices_wp[edge_end].map(|n| n as f64),
                    normal_in,
                );
                let orientation = dir.dot(plane_normal);

                let distance = point_line_distance(
                    pos,
                    vertices_wp[edge_start].map(|n| n as f64),
                    vertices_wp[edge_end].map(|n| n as f64),
                )
                .1;
                if orientation < -eps && distance < min_distance && distance < eps {
                    min_distance = distance;
                    is_on_edge = Some(((edge_start, edge_end), opposite_vertex));
                }
            }
            is_on_edge
        };

        if let Some(vertex) = is_on_vertex {
            let edges = match vertex {
                0 => [
                    (vertices_wp[0], vertices_wp[1]),
                    (vertices_wp[2], vertices_wp[0]),
                ],
                1 => [
                    (vertices_wp[0], vertices_wp[1]),
                    (vertices_wp[1], vertices_wp[2]),
                ],
                2 => [
                    (vertices_wp[1], vertices_wp[2]),
                    (vertices_wp[2], vertices_wp[0]),
                ],
                _ => panic!(),
            };
            let edge_opposite_vertex = match vertex {
                0 => [vertices_wp[2], vertices_wp[1]],
                1 => [vertices_wp[2], vertices_wp[0]],
                2 => [vertices_wp[0], vertices_wp[1]],
                _ => panic!(),
            };

            let plane_normals = [
                oriented_plane_normal(
                    edge_opposite_vertex[0].map(|n| n as f64),
                    edges[0].0.map(|n| n as f64),
                    edges[0].1.map(|n| n as f64),
                    normal_in,
                ),
                oriented_plane_normal(
                    edge_opposite_vertex[1].map(|n| n as f64),
                    edges[1].0.map(|n| n as f64),
                    edges[1].1.map(|n| n as f64),
                    normal_in,
                ),
            ];

            let orientation = [dir.dot(plane_normals[0]), dir.dot(plane_normals[1])];

            if orientation[0] < -eps || orientation[1] < -eps {
                let vertex_wp = vertices_wp[vertex];

                for i in merged_vertices[&vertex_wp.map(|n| n.to_bits())]
                    .iter()
                    .copied()
                {
                    if i == triangle {
                        continue;
                    }
                    let i = i as usize;
                    let opposite_vertices_wp: [Point3<f32>; 3] = std::array::from_fn(|j| {
                        Point3::new(
                            vertices_world_positions[[indices[[i, j]] as usize, 0]],
                            vertices_world_positions[[indices[[i, j]] as usize, 1]],
                            vertices_world_positions[[indices[[i, j]] as usize, 2]],
                        )
                    });
                    let mut shared_vertex = None;
                    for j in 0..3 {
                        if vertex_wp == opposite_vertices_wp[j] {
                            shared_vertex = Some(j);
                            break;
                        }
                    }

                    let shared_vertex = match shared_vertex {
                        Some(shared_vertex) => shared_vertex,
                        None => continue,
                    };

                    let normal_out = (opposite_vertices_wp[1] - opposite_vertices_wp[0])
                        .cross(opposite_vertices_wp[2] - opposite_vertices_wp[0])
                        .map(|n| n as f64)
                        .normalize();

                    let opposite_edges = match shared_vertex {
                        0 => [
                            (opposite_vertices_wp[0], opposite_vertices_wp[1]),
                            (opposite_vertices_wp[2], opposite_vertices_wp[0]),
                        ],
                        1 => [
                            (opposite_vertices_wp[0], opposite_vertices_wp[1]),
                            (opposite_vertices_wp[1], opposite_vertices_wp[2]),
                        ],
                        2 => [
                            (opposite_vertices_wp[1], opposite_vertices_wp[2]),
                            (opposite_vertices_wp[2], opposite_vertices_wp[0]),
                        ],
                        _ => panic!(),
                    };
                    let opposite_edge_opposite_vertex = match shared_vertex {
                        0 => [opposite_vertices_wp[2], opposite_vertices_wp[1]],
                        1 => [opposite_vertices_wp[2], opposite_vertices_wp[0]],
                        2 => [opposite_vertices_wp[0], opposite_vertices_wp[1]],
                        _ => panic!(),
                    };

                    let opposite_plane_normals = [
                        oriented_plane_normal(
                            opposite_edge_opposite_vertex[0].map(|n| n as f64),
                            opposite_edges[0].0.map(|n| n as f64),
                            opposite_edges[0].1.map(|n| n as f64),
                            normal_out,
                        ),
                        oriented_plane_normal(
                            opposite_edge_opposite_vertex[1].map(|n| n as f64),
                            opposite_edges[1].0.map(|n| n as f64),
                            opposite_edges[1].1.map(|n| n as f64),
                            normal_out,
                        ),
                    ];

                    let rot = rotation_from_two_vectors(normal_in, normal_out);
                    let opposite_dir = rot * dir;

                    let opposite_orientation = [
                        opposite_dir.dot(opposite_plane_normals[0]),
                        opposite_dir.dot(opposite_plane_normals[1]),
                    ];

                    if opposite_orientation[0] >= 0.0 && opposite_orientation[1] >= 0.0 {
                        let new_pos = vertex_wp.map(|n| n as f64);
                        length -= if new_pos == pos {
                            0.0
                        } else {
                            pos.distance(new_pos)
                        };
                        pos = new_pos;
                        dir = opposite_dir;
                        triangle = i as i32;
                        continue 'primary;
                    }
                }

                // this is only reachable if we could not find any opposite triangle
                length = 0.0;
                successful = false;
                break 'primary;
            }
        }
        if let Some(((edge_start, edge_end), opposite_vertex)) = is_on_edge {
            let plane_normal = oriented_plane_normal(
                vertices_wp[opposite_vertex].map(|n| n as f64),
                vertices_wp[edge_start].map(|n| n as f64),
                vertices_wp[edge_end].map(|n| n as f64),
                normal_in,
            );

            let orientation = dir.dot(plane_normal);
            if orientation < -eps {
                // edge_start is edge index
                let next_triangle = triangle_edge_neighborhoods[triangle as usize * 3 + edge_start];
                if next_triangle.0 >= 0 {
                    let next_vertices_wp: [Point3<f32>; 3] = std::array::from_fn(|i| {
                        Point3::new(
                            vertices_world_positions
                                [[indices[[next_triangle.0 as usize, i]] as usize, 0]],
                            vertices_world_positions
                                [[indices[[next_triangle.0 as usize, i]] as usize, 1]],
                            vertices_world_positions
                                [[indices[[next_triangle.0 as usize, i]] as usize, 2]],
                        )
                    });
                    let normal_out = (next_vertices_wp[1] - next_vertices_wp[0])
                        .cross(next_vertices_wp[2] - next_vertices_wp[0])
                        .map(|n| n as f64)
                        .normalize();

                    let rot = rotation_from_two_vectors(normal_in, normal_out);
                    dir = rot * dir;

                    let new_pos = point_line_distance(
                        pos,
                        vertices_wp[edge_start].map(|n| n as f64),
                        vertices_wp[edge_end].map(|n| n as f64),
                    )
                    .0;
                    length -= if pos == new_pos {
                        0.0
                    } else {
                        pos.distance(new_pos)
                    };
                    pos = new_pos;
                    triangle = next_triangle.0;
                    continue 'primary;
                } else {
                    length = 0.0;
                    successful = false;
                    break 'primary;
                }
            }
        }

        // at this point we know that the dir is facing inward the current triangle or it follows one of the edges
        // first we detect if we are going alongside an edge

        // if we are on a vertex, we are on two edges
        let mut dir_on_edge = None;
        if let Some(vertex) = is_on_vertex {
            let edges = match vertex {
                0 => [
                    (vertices_wp[0], vertices_wp[1]),
                    (vertices_wp[2], vertices_wp[0]),
                ],
                1 => [
                    (vertices_wp[0], vertices_wp[1]),
                    (vertices_wp[1], vertices_wp[2]),
                ],
                2 => [
                    (vertices_wp[1], vertices_wp[2]),
                    (vertices_wp[2], vertices_wp[0]),
                ],
                _ => panic!(),
            };

            let edge_directions = [
                (edges[0].1 - edges[0].0).normalize(),
                (edges[1].1 - edges[1].0).normalize(),
            ];

            let dir_similarities = [
                dir.dot(edge_directions[0].map(|n| n as f64)).abs(),
                dir.dot(edge_directions[1].map(|n| n as f64)).abs(),
            ];

            let max_similarity_idx = if dir_similarities[0] >= dir_similarities[1] {
                0
            } else {
                1
            };

            if dir_similarities[max_similarity_idx] > 1.0 - eps {
                dir_on_edge = Some(edges[max_similarity_idx]);
            }
        } else if let Some(((edge_start, edge_end), _)) = is_on_edge {
            let edge_direction = (vertices_wp[edge_end] - vertices_wp[edge_start])
                .map(|n| n as f64)
                .normalize();
            let dir_similarity = dir.dot(edge_direction).abs();

            if dir_similarity > 1.0 - eps {
                dir_on_edge = Some((vertices_wp[edge_start], vertices_wp[edge_end]));
            }
        }

        if let Some((edge_start, edge_end)) = dir_on_edge {
            let edge_direction = (edge_end - edge_start).map(|n| n as f64).normalize();

            let orientation = edge_direction.dot(dir);

            let target_vertex = if orientation > eps {
                edge_end.map(|n| n as f64)
            } else {
                edge_start.map(|n| n as f64)
            };

            let distance = pos.distance(target_vertex);

            if length < distance {
                pos += length * ((target_vertex - pos).normalize());
                length = 0.0;
                successful = true;
                break 'primary;
            }
            if length == distance {
                pos = target_vertex;
                length = 0.0;
                successful = true;
                break 'primary;
            } else {
                pos = target_vertex;
                length -= distance;
                continue 'primary;
            }
        }

        let mut edges = [
            Some((vertices_wp[0], vertices_wp[1], vertices_wp[2])),
            Some((vertices_wp[1], vertices_wp[2], vertices_wp[0])),
            Some((vertices_wp[2], vertices_wp[0], vertices_wp[1])),
        ];

        if let Some(vertex) = is_on_vertex {
            match vertex {
                0 => {
                    edges[0] = None;
                    edges[2] = None;
                }
                1 => {
                    edges[0] = None;
                    edges[1] = None;
                }
                2 => {
                    edges[1] = None;
                    edges[2] = None;
                }
                _ => panic!(),
            }
        }
        if let Some(((edge_start, _), _)) = is_on_edge {
            edges[edge_start] = None;
        }

        let mut best_point_in = None;
        let mut best_distance = std::f64::INFINITY;
        for (edge_start, edge_end, opposite_vertex) in edges.iter().copied().filter_map(|e| e) {
            let (a, _, distance) = line_line_distance(
                pos,
                pos + dir,
                edge_start.map(|n| n as f64),
                edge_end.map(|n| n as f64),
                true,
                false,
                true,
                true,
            );

            let plane_normal = oriented_plane_normal(
                opposite_vertex.map(|n| n as f64),
                edge_start.map(|n| n as f64),
                edge_end.map(|n| n as f64),
                normal_in,
            );
            let orientation = dir.dot(plane_normal);

            if a.x.is_finite() && orientation < -eps && distance < best_distance {
                best_point_in = Some(a);
                best_distance = distance;
            }
        }

        let go_back = if best_point_in.is_some() {
            let best_point_in = best_point_in.unwrap();

            let distance_to_intersection = pos.distance(best_point_in);

            if distance_to_intersection > 0.0 {
                if length < distance_to_intersection {
                    let new_pos = pos + length * ((best_point_in - pos).normalize());
                    pos = new_pos;
                    length = 0.0;
                    successful = true;
                    break 'primary;
                } else if length == distance_to_intersection {
                    pos = best_point_in;
                    length = 0.0;
                    successful = true;
                    break 'primary;
                }

                pos = best_point_in;
                length -= distance_to_intersection;
                continue 'primary;
            } else {
                true
            }
        } else {
            true
        };

        if go_back {
            let edges = [
                (vertices_wp[0], vertices_wp[1], vertices_wp[2]),
                (vertices_wp[1], vertices_wp[2], vertices_wp[0]),
                (vertices_wp[2], vertices_wp[0], vertices_wp[1]),
            ];

            let mut best_opposite_point_in = None;
            let mut best_opposite_distance = std::f64::INFINITY;
            for (edge_start, edge_end, _) in edges.iter().copied() {
                let (a, _, distance) = line_line_distance(
                    pos,
                    pos - dir,
                    edge_start.map(|n| n as f64),
                    edge_end.map(|n| n as f64),
                    true,
                    false,
                    true,
                    true,
                );
                if distance < best_opposite_distance {
                    best_opposite_point_in = Some(a);
                    best_opposite_distance = distance;
                }
            }

            if best_opposite_point_in.is_none() {
                let mut best_opposite_distance = std::f64::INFINITY;
                for (edge_start, edge_end, _) in edges.iter().copied() {
                    let (a, _, distance) = line_line_distance(
                        pos,
                        pos - dir,
                        edge_start.map(|n| n as f64),
                        edge_end.map(|n| n as f64),
                        true,
                        false,
                        true,
                        true,
                    );
                    if distance < best_opposite_distance {
                        best_opposite_point_in = Some(a);
                        best_opposite_distance = distance;
                    }
                }
            }

            let best_opposite_point_in =
                if let Some(best_opposite_point_in) = best_opposite_point_in {
                    best_opposite_point_in
                } else {
                    successful = false;
                    break 'primary;
                };

            pos = best_opposite_point_in;
        }
    }

    (triangle, pos, dir, successful)
}
