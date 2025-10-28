use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use cgmath::prelude::*;
use cgmath::{dot, Point2, Point3, Vector2, Vector3};

use ndarray::prelude::*;
use ndarray::{Array, Ix2};

use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};

use ordered_float::OrderedFloat;

use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::Python;

use serde::{Deserialize, Serialize};

use crate::float::Float;
use crate::line::Line;
use crate::surface_sampling::{trace_surface_line, trace_surface_line_with_samples};
use crate::triangle::Triangle;
use crate::utils::*;

// for visualising textures with an external program
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mesh {
    indices: Vec<u32>,
    world_coords: Vec<f32>,
    uvs: Vec<f32>,
    textures: HashMap<String, (usize, usize, Vec<u8>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PoolPoints {
    world_coords: Vec<f32>,
    texture_coords: Vec<i32>,
    triangle_ids: Vec<i32>,
    barycentric_coords: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Group<F: Float> {
    id: usize,
    position: Point3<F>,
    texels: Vec<Point2<usize>>,
}

fn uv_inject_triangle_to_candidate_texture<F: Float>(
    triangle: &Triangle<F>,
    triangle_candidates: &mut Array<Vec<usize>, Ix2>,
    resolution: Vector2<F>,
) {
    let bb_min = Point2::new(
        (triangle.uvs[0]
            .x
            .min(triangle.uvs[1].x)
            .min(triangle.uvs[2].x)
            * resolution.x)
            .floor(),
        (triangle.uvs[0]
            .y
            .min(triangle.uvs[1].y)
            .min(triangle.uvs[2].y)
            * resolution.y)
            .floor(),
    );
    let bb_max = Point2::new(
        (triangle.uvs[0]
            .x
            .max(triangle.uvs[1].x)
            .max(triangle.uvs[2].x)
            * resolution.x)
            .ceil(),
        (triangle.uvs[0]
            .y
            .max(triangle.uvs[1].y)
            .max(triangle.uvs[2].y)
            * resolution.y)
            .ceil(),
    );

    let bb_min = Point2::new(
        bb_min.x.max(F::zero()).min(resolution.x - F::one()),
        bb_min.y.max(F::zero()).min(resolution.y - F::one()),
    );

    let bb_max = Point2::new(
        bb_max.x.max(F::zero()).min(resolution.x - F::one()),
        bb_max.y.max(F::zero()).min(resolution.y - F::one()),
    );

    let bb_min: Point2<usize> = bb_min.cast().unwrap();
    let bb_max: Point2<usize> = bb_max.cast().unwrap();

    for x in bb_min.x..bb_max.x {
        for y in bb_min.y..bb_max.y {
            let uv = Point2::new(
                F::from(x).unwrap() / resolution.x
                    + (F::one() / (resolution.x * F::from(2.0).unwrap())),
                F::from(y).unwrap() / resolution.y
                    + (F::one() / (resolution.y * F::from(2.0).unwrap())),
            );

            if let Some(barycentric) = triangle.barycentric_of_uv(uv) {
                if Triangle::barycentric_is_inside(barycentric) {
                    triangle_candidates[[x, y]].push(triangle.id);
                    continue;
                }
            }

            let triangle_lines = [
                Line::new(triangle.uvs[0], triangle.uvs[1]),
                Line::new(triangle.uvs[1], triangle.uvs[2]),
                Line::new(triangle.uvs[2], triangle.uvs[0]),
            ];

            let grid_points = [
                Point2::new(
                    F::from(x).unwrap() / resolution.x,
                    F::from(y).unwrap() / resolution.y,
                ),
                Point2::new(
                    F::from(x + 1).unwrap() / resolution.x,
                    F::from(y).unwrap() / resolution.y,
                ),
                Point2::new(
                    F::from(x + 1).unwrap() / resolution.x,
                    F::from(y + 1).unwrap() / resolution.y,
                ),
                Point2::new(
                    F::from(x).unwrap() / resolution.x,
                    F::from(y + 1).unwrap() / resolution.y,
                ),
            ];
            let grid_lines = [
                Line::new(grid_points[0], grid_points[1]),
                Line::new(grid_points[1], grid_points[2]),
                Line::new(grid_points[2], grid_points[3]),
                Line::new(grid_points[3], grid_points[0]),
            ];

            let mut found_candidate = false;
            'line_check: for triangle_line in triangle_lines.iter() {
                for grid_line in grid_lines.iter() {
                    if triangle_line.intersects_with_endpoints(grid_line) {
                        triangle_candidates[[x, y]].push(triangle.id);
                        found_candidate = true;
                        break 'line_check;
                    }
                }
            }
            if found_candidate {
                continue;
            }

            let bb_uv_min = Point2::new(
                F::from(x).unwrap() / resolution.x,
                F::from(y).unwrap() / resolution.y,
            );
            let bb_uv_max = Point2::new(
                F::from(x + 1).unwrap() / resolution.x,
                F::from(y + 1).unwrap() / resolution.y,
            );

            let mut counter = 0;
            for uv in triangle.uvs.iter() {
                if bb_uv_min.x <= uv.x
                    && uv.x <= bb_uv_max.x
                    && bb_uv_min.y <= uv.y
                    && uv.y <= bb_uv_max.y
                {
                    counter += 1;
                }
            }
            if counter > 0 {
                triangle_candidates[[x, y]].push(triangle.id);
            }
        }
    }
}

fn uv_inject_triangle_world_positions<F: Float>(
    triangles: &[Triangle<F>],
    resolution: Vector2<usize>,
) -> (
    Array<Option<Point3<F>>, Ix2>,
    Array<Option<(usize, Vector3<F>)>, Ix2>,
) {
    let mut triangle_candidates =
        Array::<Vec<usize>, Ix2>::from_elem(Into::<[usize; 2]>::into(resolution), vec![]);
    let resolution_float: Vector2<F> = resolution.cast().unwrap();

    for triangle in triangles.iter() {
        uv_inject_triangle_to_candidate_texture(
            triangle,
            &mut triangle_candidates,
            resolution_float,
        );
    }

    let mut result_wp = Array::<_, Ix2>::from_elem(Into::<[usize; 2]>::into(resolution), None);
    let mut result_triangles =
        Array::<_, Ix2>::from_elem(Into::<[usize; 2]>::into(resolution), None);

    let hdx_hdy = Point2::new(
        F::one() / F::from(resolution.x * 2).unwrap(),
        F::one() / F::from(resolution.y * 2).unwrap(),
    );

    for x in 0..resolution.x {
        for y in 0..resolution.y {
            let uv = Point2::new(
                F::from(x).unwrap() / resolution_float.x + hdx_hdy.x,
                F::from(y).unwrap() / resolution_float.y + hdx_hdy.y,
            );

            if triangle_candidates[[x, y]].is_empty() {
                continue;
            }

            for triangle_id in triangle_candidates[[x, y]].iter().copied() {
                let triangle = triangles[triangle_id];
                if let Some(barycentric) = triangle.barycentric_of_uv(uv) {
                    if Triangle::barycentric_is_inside(barycentric) {
                        result_wp[[x, y]] = Some(triangle.world_position(barycentric));
                        result_triangles[[x, y]] = Some((triangle_id, barycentric));
                    }
                }
            }

            if result_wp[[x, y]].is_some() {
                continue;
            }

            let mut closest_distance = F::infinity();
            let mut closest_triangle_id = None;
            let mut closest_uv_point = Point2::new(F::zero(), F::zero());
            for triangle_id in triangle_candidates[[x, y]].iter().copied() {
                let triangle = triangles[triangle_id];

                if let Some((triangle_closest_uv_point, distance)) = triangle.closest_uv_point(uv) {
                    if distance < closest_distance {
                        closest_triangle_id = Some(triangle_id);
                        closest_uv_point = triangle_closest_uv_point;
                        closest_distance = distance;
                    }
                }
            }

            if let Some(closest_triangle_id) = closest_triangle_id {
                let triangle = triangles[closest_triangle_id];

                // floating point errors correction
                let direction = closest_uv_point - uv;
                let direction = direction * F::from(0.001).unwrap();
                let closest_uv_point = closest_uv_point + direction;

                if let Some(barycentric) = triangle.barycentric_of_uv(closest_uv_point) {
                    result_wp[[x, y]] = Some(triangle.world_position(barycentric));
                    result_triangles[[x, y]] = Some((closest_triangle_id, barycentric));
                }
            }
        }
    }

    for x in 0..resolution.x {
        for y in 0..resolution.y {
            if let Some((triangle_id, barycentric)) = result_triangles[[x, y]] {
                result_triangles[[x, y]] =
                    Some((triangle_id, Triangle::barycentric_snap_inside(barycentric)));
            }
        }
    }

    (result_wp, result_triangles)
}

fn compute_normals<F: Float>(
    world_positions: &[Point3<F>],
    indices: &[[usize; 3]],
) -> Vec<Vector3<F>> {
    let mut normals = vec![Vector3::new(F::zero(), F::zero(), F::zero()); world_positions.len()];
    let mut merged_vertices = HashMap::<Point3<F::BitType>, (Vec<Vector3<F>>, Vec<usize>)>::new();
    for (idx, world_position) in world_positions.iter().enumerate() {
        let bit_vertex = world_position.map(|n| n.to_bits());
        merged_vertices
            .entry(bit_vertex)
            .and_modify(|m: &mut (Vec<Vector3<F>>, Vec<usize>)| m.1.push(idx))
            .or_insert_with(|| (vec![], vec![idx]));
    }

    for triangle in indices.iter() {
        let a = world_positions[triangle[0]];
        let b = world_positions[triangle[1]];
        let c = world_positions[triangle[2]];

        let area_scaled_normal = (b - a).cross(c - a);

        normals[triangle[0]] += area_scaled_normal;
        normals[triangle[1]] += area_scaled_normal;
        normals[triangle[2]] += area_scaled_normal;

        for vertex in [a, b, c].iter() {
            let vertex_bits = vertex.map(|n| n.to_bits());
            if let Some(m) = merged_vertices.get_mut(&vertex_bits) {
                m.0.push(area_scaled_normal);
            }
        }
    }

    for normal in normals.iter_mut() {
        *normal = normal.normalize();
    }

    normals
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct MeshLayer {
    mesh_id: usize,
    layer: usize,
}

#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Tmscnn {
    groups: HashMap<MeshLayer, Vec<Group<f32>>>,
    progressive_pooling_reductions: Vec<f32>,

    pooling_forward_mapping: HashMap<MeshLayer, Vec<usize>>,
    pooling_backward_mapping: HashMap<MeshLayer, Vec<Vec<usize>>>,
}

#[pymethods]
impl Tmscnn {
    #[new]
    pub fn new() -> Tmscnn {
        Tmscnn {
            groups: HashMap::new(),
            progressive_pooling_reductions: Vec::new(),

            pooling_forward_mapping: HashMap::new(),
            pooling_backward_mapping: HashMap::new(),
        }
    }

    pub fn load_obj(&self, py: Python, path: &str) -> PyResult<PyObject> {
        let result = PyDict::new(py);

        let (models, _) = tobj::load_obj(&path, &tobj::LoadOptions::default()).unwrap();

        let model = models.iter().next().unwrap();

        let mesh = &model.mesh;

        let n_triangles = mesh.indices.len() / 3;
        let has_uvs = mesh.texcoord_indices.len() > 0;

        let world_positions = PyArray2::<f32>::zeros(py, (mesh.positions.len() / 3, 3), false);

        for i in 0..world_positions.shape()[0] {
            unsafe {
                *world_positions.uget_mut([i, 0]) = mesh.positions[i * 3];
                *world_positions.uget_mut([i, 1]) = mesh.positions[i * 3 + 1];
                *world_positions.uget_mut([i, 2]) = mesh.positions[i * 3 + 2];
            }
        }

        let world_position_indices = PyArray2::<i32>::zeros(py, (n_triangles, 3), false);

        for i in 0..n_triangles {
            unsafe {
                *world_position_indices.uget_mut([i, 0]) = mesh.indices[i * 3] as i32;
                *world_position_indices.uget_mut([i, 1]) = mesh.indices[i * 3 + 1] as i32;
                *world_position_indices.uget_mut([i, 2]) = mesh.indices[i * 3 + 2] as i32;
            }
        }

        result
            .set_item("world_positions".to_object(py), world_positions)
            .unwrap();
        result
            .set_item(
                "world_position_indices".to_object(py),
                world_position_indices,
            )
            .unwrap();

        if has_uvs {
            let uvs = PyArray2::<f32>::zeros(py, (mesh.texcoords.len() / 2, 2), false);

            for i in 0..uvs.shape()[0] {
                unsafe {
                    *uvs.uget_mut([i, 0]) = 1.0 - mesh.texcoords[i * 2 + 1];
                    *uvs.uget_mut([i, 1]) = mesh.texcoords[i * 2];
                }
            }

            let uv_indices = PyArray2::<i32>::zeros(py, (n_triangles, 3), false);

            for i in 0..n_triangles {
                unsafe {
                    *uv_indices.uget_mut([i, 0]) = mesh.texcoord_indices[i * 3] as i32;
                    *uv_indices.uget_mut([i, 1]) = mesh.texcoord_indices[i * 3 + 1] as i32;
                    *uv_indices.uget_mut([i, 2]) = mesh.texcoord_indices[i * 3 + 2] as i32;
                }
            }

            result.set_item("uvs".to_object(py), uvs).unwrap();
            result
                .set_item("uv_indices".to_object(py), uv_indices)
                .unwrap();
        }

        Ok(result.into())
    }

    pub fn save_obj(
        &self,
        path: &str,
        wps_py: &PyArray2<f32>,
        uvs_py: &PyArray2<f32>,
        indices_py: &PyArray2<i32>,
    ) {
        let mut wps = Vec::with_capacity(wps_py.shape()[0]);
        let mut uvs = Vec::with_capacity(uvs_py.shape()[0]);
        let mut indices = Vec::with_capacity(indices_py.shape()[0]);

        for i in 0..wps_py.shape()[0] {
            unsafe {
                wps.push(Point3::new(
                    OrderedFloat(*wps_py.uget([i, 0])),
                    OrderedFloat(*wps_py.uget([i, 1])),
                    OrderedFloat(*wps_py.uget([i, 2])),
                ));
                uvs.push(Point2::new(*uvs_py.uget([i, 0]), *uvs_py.uget([i, 1])));
            }
        }
        for i in 0..indices_py.shape()[0] {
            unsafe {
                indices.push(Point3::new(
                    *indices_py.uget([i, 0]),
                    *indices_py.uget([i, 1]),
                    *indices_py.uget([i, 2]),
                ));
            }
        }

        let mut collapsed_wps = vec![];
        let mut wp_mapping = HashMap::<Point3<OrderedFloat<f32>>, usize>::new();

        for wp in wps.iter().copied() {
            if !wp_mapping.contains_key(&wp) {
                let idx = collapsed_wps.len();
                wp_mapping.insert(wp, idx);
                collapsed_wps.push(wp.map(|n| n));
            }
        }

        let file = std::fs::File::create(path).unwrap();
        let mut writer = std::io::BufWriter::new(file);

        use std::io::Write;
        for wp in collapsed_wps.iter() {
            write!(
                writer,
                "v {} {} {}\n",
                wp.x.0 as f64, wp.y.0 as f64, wp.z.0 as f64
            )
            .unwrap();
        }
        for uv in uvs.iter() {
            write!(writer, "vt {} {}\n", uv.y as f64, 1.0 - uv.x as f64).unwrap();
        }
        for triangle in indices.iter() {
            let triangle_collapsed = triangle.map(|i| wp_mapping[&wps[i as usize]]);
            assert!(wps[triangle[0] as usize] == collapsed_wps[triangle_collapsed[0]]);
            assert!(wps[triangle[1] as usize] == collapsed_wps[triangle_collapsed[1]]);
            assert!(wps[triangle[2] as usize] == collapsed_wps[triangle_collapsed[2]]);

            write!(
                writer,
                "f {}/{} {}/{} {}/{}\n",
                triangle_collapsed[0] + 1,
                triangle[0] + 1,
                triangle_collapsed[1] + 1,
                triangle[1] + 1,
                triangle_collapsed[2] + 1,
                triangle[2] + 1
            )
            .unwrap();
        }
    }

    pub fn compute_normals(
        &self,
        py: Python,
        vertices: &PyArray2<f32>,
        indices: &PyArray2<i32>,
    ) -> PyResult<PyObject> {
        let vertices = vertices.readonly();
        let indices = indices.readonly();

        let vertices = (0..vertices.shape()[0])
            .map(|idx| {
                Point3::new(
                    vertices.get_owned([idx, 0]).unwrap(),
                    vertices.get_owned([idx, 1]).unwrap(),
                    vertices.get_owned([idx, 2]).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let indices = (0..indices.shape()[0])
            .map(|idx| {
                [
                    indices.get_owned([idx, 0]).unwrap().try_into().unwrap(),
                    indices.get_owned([idx, 1]).unwrap().try_into().unwrap(),
                    indices.get_owned([idx, 2]).unwrap().try_into().unwrap(),
                ]
            })
            .collect::<Vec<[usize; 3]>>();

        let normals = compute_normals(&vertices, &indices);

        let normals_py = PyArray2::<f32>::zeros(py, (vertices.len(), 3), false);

        for idx in 0..vertices.len() {
            unsafe {
                *normals_py.uget_mut([idx, 0]) = normals[idx].x;
                *normals_py.uget_mut([idx, 1]) = normals[idx].y;
                *normals_py.uget_mut([idx, 2]) = normals[idx].z;
            }
        }

        Ok(normals_py.into())
    }

    pub fn textures_to_flattened_groups_forward_pass(
        &self,
        py: Python,
        mesh_ids: &PyArray2<i32>,
        textures: &PyArray4<f32>,
    ) -> PyResult<PyObject> {
        let mesh_ids_readonly = mesh_ids.readonly();
        let textures_readonly = textures.readonly();

        let mesh_ids = mesh_ids_readonly.as_array();
        let textures = textures_readonly.as_array();

        let mut n_groups = 0;
        let n_channels = textures.raw_dim()[3];

        for batch in 0..mesh_ids.raw_dim()[0] {
            let mesh_layer = MeshLayer {
                mesh_id: *mesh_ids.get([batch, 0]).unwrap() as usize,
                layer: 0,
            };
            let groups = &self.groups[&mesh_layer];
            n_groups += groups.len();
        }

        let flattened_groups = PyArray2::<f32>::zeros(py, (n_groups, n_channels), false);

        let mut offset = 0;
        for batch in 0..mesh_ids.raw_dim()[0] {
            let mesh_layer = MeshLayer {
                mesh_id: *mesh_ids.get([batch, 0]).unwrap() as usize,
                layer: 0,
            };
            let groups = &self.groups[&mesh_layer];

            for (idx, group) in groups.iter().enumerate() {
                for channel in 0..n_channels {
                    unsafe {
                        *flattened_groups.uget_mut([idx + offset, channel]) =
                            textures[[batch, group.texels[0].x, group.texels[0].y, channel]];
                    }
                }
            }

            offset += groups.len();
        }

        Ok(flattened_groups.into())
    }

    pub fn textures_to_flattened_groups_backward_pass(
        &self,
        textures_grad: &PyArray4<f32>,
        mesh_ids: &PyArray2<i32>,
        flattened_groups_grad: &PyArray2<f32>,
    ) {
        let mut textures_grad = unsafe { textures_grad.as_array_mut() };

        let mesh_ids_readonly = mesh_ids.readonly();
        let flattened_groups_grad_readonly = flattened_groups_grad.readonly();

        let mesh_ids = mesh_ids_readonly.as_array();
        let flattened_groups_grad = flattened_groups_grad_readonly.as_array();

        let n_channels = flattened_groups_grad.raw_dim()[1];

        let mut offset = 0;
        for batch in 0..mesh_ids.raw_dim()[0] {
            let mesh_layer = MeshLayer {
                mesh_id: *mesh_ids.get([batch, 0]).unwrap() as usize,
                layer: 0,
            };
            let groups = &self.groups[&mesh_layer];

            for (group_idx, group) in groups.iter().enumerate() {
                let flattened_group_grad_idx = group_idx + offset;
                for texel in group.texels.iter() {
                    for channel in 0..n_channels {
                        textures_grad[[batch, texel.x, texel.y, channel]] =
                            flattened_groups_grad[[flattened_group_grad_idx, channel]];
                    }
                }
            }

            offset += groups.len();
        }
    }

    pub fn create_plane_neighborhoods(
        &mut self,
        py: Python,
        mesh_id: usize,
        resolution: &PyTuple,
        n_layers: usize,
    ) -> PyResult<PyObject> {
        let resolution: (usize, usize) = resolution.extract().unwrap();
        let mut resolution = Vector2::new(resolution.0, resolution.1);
        let original_resolution = resolution;

        let linear_idx = |x, y, height| x * height + y;

        let result = PyDict::new(py);

        let sources_nearest_layers = PyList::empty(py);
        let sources_linear_layers = PyList::empty(py);
        let scaling_factors_layers = PyList::empty(py);
        let backward_correction_terms_layers = PyList::empty(py);
        let backward_correction_terms_nearest_layers = PyList::empty(py);
        let pooling_spans_layers = PyList::empty(py);
        let pooling_indices_layers = PyList::empty(py);

        let group_textures_layers = PyList::empty(py);
        let group_centers = PyList::empty(py);

        let tangent_texture = PyArray3::<f32>::zeros(py, (resolution.x, resolution.y, 3), false);
        let tangent_discontinuity_texture =
            PyArray2::<f32>::zeros(py, (resolution.x, resolution.y), false);
        let tangent_discontinuity_mask =
            PyArray2::<f32>::zeros(py, (resolution.x, resolution.y), false);

        for x in 0..resolution.x {
            for y in 0..resolution.y {
                unsafe {
                    *tangent_texture.uget_mut([x, y, 0]) = 0.0;
                    *tangent_texture.uget_mut([x, y, 0]) = 1.0;
                    *tangent_texture.uget_mut([x, y, 0]) = 0.0;

                    *tangent_discontinuity_texture.uget_mut([x, y]) = 1.0;

                    *tangent_discontinuity_mask.uget_mut([x, y]) = 1.0;
                }
            }
        }

        let mut groups = vec![];
        for x in 0..resolution.x {
            for y in 0..resolution.y {
                groups.push(Group {
                    id: linear_idx(x, y, resolution.y),
                    position: Point3::new(x as f32, y as f32, 0.0),
                    texels: vec![Point2::new(x, y)],
                })
            }
        }

        for layer in 0..n_layers {
            let conv_terms_nearest_len = groups.len() * 3 * 3;
            let conv_terms_linear_len = conv_terms_nearest_len * 4;
            let sources_nearest = PyArray1::<i32>::new(py, conv_terms_nearest_len, false);
            let sources_linear = PyArray1::<i32>::new(py, conv_terms_linear_len, false);
            let scaling_factors = PyArray1::<f32>::new(py, conv_terms_linear_len, false);
            let backward_correction_terms = PyArray1::<f32>::new(py, groups.len(), false);
            let backward_correction_terms_nearest = PyArray1::<f32>::new(py, groups.len(), false);

            let mesh_layer = MeshLayer { mesh_id, layer };

            for x in 0..resolution.x {
                for y in 0..resolution.y {
                    for kernel_x in 0..3 {
                        let dx = kernel_x as i32 - 1;
                        for kernel_y in 0..3 {
                            let dy = kernel_y as i32 - 1;

                            let weight_id = kernel_x * 3 + kernel_y;

                            let x = x as i32;
                            let y = y as i32;
                            let source = if x + dx >= 0
                                && x + dx < resolution.x as i32
                                && y + dy >= 0
                                && y + dy < resolution.y as i32
                            {
                                linear_idx((x + dx) as usize, (y + dy) as usize, resolution.y)
                                    as i32
                            } else {
                                linear_idx(x as usize, y as usize, resolution.y) as i32
                            };

                            let term_id_nearest =
                                linear_idx(x as usize, y as usize, resolution.y) * 9 + weight_id;
                            unsafe {
                                *sources_nearest.uget_mut([term_id_nearest]) = source;
                            }

                            for i in 0..4 {
                                let term_id_linear =
                                    linear_idx(x as usize, y as usize, resolution.y) * 9 * 4
                                        + weight_id * 4
                                        + i;

                                unsafe {
                                    *sources_linear.uget_mut([term_id_linear]) = source;
                                    if i == 0 {
                                        *scaling_factors.uget_mut([term_id_linear]) = 1.0;
                                    } else {
                                        *scaling_factors.uget_mut([term_id_linear]) = 0.0;
                                    }
                                }
                            }
                        }
                        unsafe {
                            *backward_correction_terms.uget_mut([linear_idx(x, y, resolution.y)]) =
                                9.0;
                            *backward_correction_terms_nearest.uget_mut([linear_idx(
                                x,
                                y,
                                resolution.y,
                            )]) = 9.0;
                        }
                    }
                }
            }

            let group_texture_py =
                PyArray2::<i32>::new(py, (original_resolution.x, original_resolution.y), false);

            self.groups.insert(mesh_layer, groups.clone());

            sources_nearest_layers.append(sources_nearest).unwrap();
            sources_linear_layers.append(sources_linear).unwrap();
            scaling_factors_layers.append(scaling_factors).unwrap();
            backward_correction_terms_layers
                .append(backward_correction_terms)
                .unwrap();
            backward_correction_terms_nearest_layers
                .append(backward_correction_terms_nearest)
                .unwrap();
            group_textures_layers.append(group_texture_py).unwrap();

            let new_resolution = Vector2::new((resolution.x + 1) / 2, (resolution.y + 1) / 2);

            if layer < n_layers - 1 {
                let pooling_spans =
                    PyArray1::<u32>::new(py, new_resolution.x * new_resolution.y + 1, false);
                let pooling_indices = PyArray1::<u32>::new(py, groups.len(), false);
                let mut next_index = 0;

                let mut pool_counter = 0;
                unsafe {
                    *pooling_spans.uget_mut([0]) = 0;
                }

                for x in 0..new_resolution.x {
                    for y in 0..new_resolution.y {
                        let mut pool_size = 0;
                        for dx in 0..2 {
                            for dy in 0..2 {
                                let old_x = 2 * x + dx;
                                let old_y = 2 * y + dy;

                                if old_x >= resolution.x || old_y >= resolution.y {
                                    continue;
                                }

                                pool_size += 1;

                                let old_linear_idx = linear_idx(old_x, old_y, resolution.y);
                                unsafe {
                                    *pooling_indices.uget_mut([next_index]) = old_linear_idx as u32;
                                }
                                next_index += 1;
                            }
                        }
                        unsafe {
                            *pooling_spans.uget_mut([pool_counter + 1]) =
                                *pooling_spans.uget_mut([pool_counter]) + pool_size;
                        }
                        pool_counter += 1;
                    }
                }

                pooling_spans_layers.append(pooling_spans).unwrap();
                pooling_indices_layers.append(pooling_indices).unwrap();
            }

            resolution = new_resolution;

            let scale_x = original_resolution.x / resolution.x;
            let scale_y = original_resolution.y / resolution.y;

            groups.clear();
            for x in 0..resolution.x {
                for y in 0..resolution.y {
                    groups.push(Group {
                        id: linear_idx(x, y, resolution.y),
                        position: Point3::new((x * scale_x) as f32, (y * scale_y) as f32, 0.0),
                        texels: vec![],
                    });
                }
            }
        }

        result
            .set_item("sources_nearest".to_object(py), sources_nearest_layers)
            .unwrap();
        result
            .set_item("sources_linear".to_object(py), sources_linear_layers)
            .unwrap();
        result
            .set_item("scaling_factors".to_object(py), scaling_factors_layers)
            .unwrap();
        result
            .set_item(
                "backward_correction_terms".to_object(py),
                backward_correction_terms_layers,
            )
            .unwrap();
        result
            .set_item(
                "backward_correction_terms_nearest".to_object(py),
                backward_correction_terms_nearest_layers,
            )
            .unwrap();
        result
            .set_item("pooling_spans".to_object(py), pooling_spans_layers)
            .unwrap();
        result
            .set_item("pooling_indices".to_object(py), pooling_indices_layers)
            .unwrap();
        result
            .set_item("group_textures".to_object(py), group_textures_layers)
            .unwrap();
        result
            .set_item("group_centers".to_object(py), group_centers)
            .unwrap();
        result
            .set_item("tangent_texture".to_object(py), tangent_texture)
            .unwrap();
        result
            .set_item(
                "tangent_discontinuity_texture".to_object(py),
                tangent_discontinuity_texture,
            )
            .unwrap();
        result
            .set_item(
                "tangent_discontinuity_mask".to_object(py),
                tangent_discontinuity_mask,
            )
            .unwrap();

        Ok(result.into())
    }

    fn create_tangent_vectors(
        &mut self,
        py: Python,
        vertices_world_positions: &PyArray2<f32>,
        indices: &PyArray2<i32>,
        tangent_primary_generator_vector: &PyTuple,
        tangent_secondary_generator_vector: &PyTuple,
    ) -> PyResult<PyObject> {
        let tangent_primary_generator_vector: (f32, f32, f32) =
            tangent_primary_generator_vector.extract().unwrap();
        let tangent_primary_generator_vector = Vector3::new(
            tangent_primary_generator_vector.0,
            tangent_primary_generator_vector.1,
            tangent_primary_generator_vector.2,
        );

        let tangent_secondary_generator_vector: (f32, f32, f32) =
            tangent_secondary_generator_vector.extract().unwrap();
        let tangent_secondary_generator_vector = Vector3::new(
            tangent_secondary_generator_vector.0,
            tangent_secondary_generator_vector.1,
            tangent_secondary_generator_vector.2,
        );

        let n_vertices = vertices_world_positions.dims()[0];
        let n_triangles = indices.dims()[0];

        let mut merged_vertices = HashMap::<Point3<u32>, (Vector3<f32>, Vec<usize>)>::new();
        for vertex_idx in 0..n_vertices {
            let vertex = Point3::new(
                vertices_world_positions.get_owned([vertex_idx, 0]).unwrap(),
                vertices_world_positions.get_owned([vertex_idx, 1]).unwrap(),
                vertices_world_positions.get_owned([vertex_idx, 2]).unwrap(),
            );
            let bit_vertex = vertex.map(|n| n.to_bits());
            merged_vertices
                .entry(bit_vertex)
                .and_modify(|m: &mut (Vector3<f32>, Vec<usize>)| m.1.push(vertex_idx))
                .or_insert_with(|| (Vector3::new(0.0, 0.0, 0.0), vec![vertex_idx]));
        }

        for triangle_idx in 0..n_triangles {
            let vertices = [
                indices.get_owned([triangle_idx, 0]).unwrap() as usize,
                indices.get_owned([triangle_idx, 1]).unwrap() as usize,
                indices.get_owned([triangle_idx, 2]).unwrap() as usize,
            ];
            let world_positions = vertices.map(|v| {
                Point3::new(
                    vertices_world_positions.get_owned([v, 0]).unwrap(),
                    vertices_world_positions.get_owned([v, 1]).unwrap(),
                    vertices_world_positions.get_owned([v, 2]).unwrap(),
                )
            });
            let area_scaled_normal = (world_positions[1] - world_positions[0])
                .cross(world_positions[2] - world_positions[1]);
            let area = area_scaled_normal.magnitude();
            let normal = area_scaled_normal.normalize();

            let p = tangent_primary_generator_vector;
            let q = tangent_secondary_generator_vector;

            let r = if (dot(normal, p) - 1.0).abs() > 0.001 {
                p
            } else {
                q
            };

            let tangent = area * normal.cross(r).normalize();

            for vertex_idx in vertices.iter().copied() {
                let vertex = Point3::new(
                    vertices_world_positions.get_owned([vertex_idx, 0]).unwrap(),
                    vertices_world_positions.get_owned([vertex_idx, 1]).unwrap(),
                    vertices_world_positions.get_owned([vertex_idx, 2]).unwrap(),
                );
                let bit_vertex = vertex.map(|n| n.to_bits());

                merged_vertices.get_mut(&bit_vertex).unwrap().0 += tangent;
            }
        }

        let tangents_py = PyArray2::<f32>::new(py, (n_vertices, 3), false);

        for (tangent, vertex_idxs) in merged_vertices.values() {
            let tangent = if tangent.magnitude() > 0.000001 {
                tangent.normalize()
            } else {
                tangent_primary_generator_vector
            };
            for vertex_idx in vertex_idxs.iter().copied() {
                unsafe {
                    *tangents_py.uget_mut([vertex_idx, 0]) = tangent.x;
                    *tangents_py.uget_mut([vertex_idx, 1]) = tangent.y;
                    *tangents_py.uget_mut([vertex_idx, 2]) = tangent.z;
                }
            }
        }

        Ok(tangents_py.into())
    }

    fn create_curved_neighborhoods(
        &mut self,
        py: Python,
        mesh_id: usize,
        vertices_world_positions: &PyArray2<f32>,
        vertices_tangents: &PyArray2<f32>,
        vertices_uvs: &PyArray2<f32>,
        wp_indices: &PyArray2<i32>,
        uv_indices: &PyArray2<i32>,
        resolution: &PyTuple,
        n_layers: usize,
    ) -> PyResult<PyObject> {
        // Convert python objects to rust structs and prepare output python objects
        let vertices_world_positions_readonly = vertices_world_positions.readonly();
        let vertices_world_positions_arr = vertices_world_positions_readonly.as_array();

        let vertices_tangents_readonly = vertices_tangents.readonly();
        let vertices_tangents_arr = vertices_tangents_readonly.as_array();

        let vertices_uvs_readonly = vertices_uvs.readonly();
        let vertices_uvs_arr = vertices_uvs_readonly.as_array();

        let wp_indices_readonly = wp_indices.readonly();
        let wp_indices_arr = wp_indices_readonly.as_array();

        let uv_indices_readonly = uv_indices.readonly();
        let uv_indices_arr = uv_indices_readonly.as_array();

        let resolution: (usize, usize) = resolution.extract().unwrap();
        let resolution = Vector2::new(resolution.0, resolution.1);

        let vertices_world_positions = (0..vertices_world_positions.shape()[0])
            .map(|idx| {
                Point3::new(
                    vertices_world_positions_arr[[idx, 0]],
                    vertices_world_positions_arr[[idx, 1]],
                    vertices_world_positions_arr[[idx, 2]],
                )
            })
            .collect::<Vec<_>>();

        let vertices_uvs = (0..vertices_uvs.shape()[0])
            .map(|idx| {
                Point2::new(
                    vertices_uvs_arr[[idx, 0]] * 0.9999999,
                    vertices_uvs_arr[[idx, 1]] * 0.9999999,
                )
            })
            .collect::<Vec<_>>();

        let n_triangles = wp_indices.shape()[0];

        let wp_indices = (0..n_triangles)
            .map(|idx| {
                [
                    wp_indices_arr[[idx, 0]].try_into().unwrap(),
                    wp_indices_arr[[idx, 1]].try_into().unwrap(),
                    wp_indices_arr[[idx, 2]].try_into().unwrap(),
                ]
            })
            .collect::<Vec<[usize; 3]>>();

        let uv_indices = (0..n_triangles)
            .map(|idx| {
                [
                    uv_indices_arr[[idx, 0]].try_into().unwrap(),
                    uv_indices_arr[[idx, 1]].try_into().unwrap(),
                    uv_indices_arr[[idx, 2]].try_into().unwrap(),
                ]
            })
            .collect::<Vec<[usize; 3]>>();

        let mut triangles = Vec::with_capacity(n_triangles);
        for i in 0..n_triangles {
            let vi1 = wp_indices[i][0];
            let vi2 = wp_indices[i][1];
            let vi3 = wp_indices[i][2];

            let uv_vi1 = uv_indices[i][0];
            let uv_vi2 = uv_indices[i][1];
            let uv_vi3 = uv_indices[i][2];

            let p1x = vertices_world_positions[vi1][0];
            let p1y = vertices_world_positions[vi1][1];
            let p1z = vertices_world_positions[vi1][2];
            let p1 = Point3::new(p1x, p1y, p1z);

            let p2x = vertices_world_positions[vi2][0];
            let p2y = vertices_world_positions[vi2][1];
            let p2z = vertices_world_positions[vi2][2];
            let p2 = Point3::new(p2x, p2y, p2z);

            let p3x = vertices_world_positions[vi3][0];
            let p3y = vertices_world_positions[vi3][1];
            let p3z = vertices_world_positions[vi3][2];
            let p3 = Point3::new(p3x, p3y, p3z);

            // we do not need vertex normals in this precomputation step
            let n1 = Vector3::new(1.0, 0.0, 0.0);
            let n2 = Vector3::new(1.0, 0.0, 0.0);
            let n3 = Vector3::new(1.0, 0.0, 0.0);

            let t1x = vertices_tangents_arr[[vi1, 0]];
            let t1y = vertices_tangents_arr[[vi1, 1]];
            let t1z = vertices_tangents_arr[[vi1, 2]];
            let t1 = Vector3::new(t1x, t1y, t1z);

            let t2x = vertices_tangents_arr[[vi2, 0]];
            let t2y = vertices_tangents_arr[[vi2, 1]];
            let t2z = vertices_tangents_arr[[vi2, 2]];
            let t2 = Vector3::new(t2x, t2y, t2z);

            let t3x = vertices_tangents_arr[[vi3, 0]];
            let t3y = vertices_tangents_arr[[vi3, 1]];
            let t3z = vertices_tangents_arr[[vi3, 2]];
            let t3 = Vector3::new(t3x, t3y, t3z);

            let uv1x = vertices_uvs[uv_vi1][0];
            let uv1y = vertices_uvs[uv_vi1][1];
            let uv1 = Point2::new(uv1x, uv1y);

            let uv2x = vertices_uvs[uv_vi2][0];
            let uv2y = vertices_uvs[uv_vi2][1];
            let uv2 = Point2::new(uv2x, uv2y);

            let uv3x = vertices_uvs[uv_vi3][0];
            let uv3y = vertices_uvs[uv_vi3][1];
            let uv3 = Point2::new(uv3x, uv3y);

            let triangle = Triangle {
                id: i,
                world_positions: [p1, p2, p3],
                normals: [n1, n2, n3],
                tangents: [t1, t2, t3],
                uvs: [uv1, uv2, uv3],
            };
            triangles.push(triangle)
        }

        let mut texel_size = compute_texel_size(&triangles, resolution);

        let (world_position_texture, triangle_texture) =
            uv_inject_triangle_world_positions(&triangles, resolution);

        // Each group contains a single texel before any pooling happens. After pooling, a single group can contain many texels
        let mut group_texture = Array::<_, Ix2>::from_elem(world_position_texture.raw_dim(), None);
        let mut groups = vec![];
        let mut group_triangles = vec![];
        for ((x, y), position) in world_position_texture.indexed_iter() {
            if let Some(position) = position {
                let group_id = groups.len();
                group_texture[[x, y]] = Some(group_id);
                groups.push(Group {
                    id: group_id,
                    position: *position,
                    texels: vec![Point2::new(x, y)],
                });
                group_triangles.push(triangle_texture[[x, y]].unwrap());
            }
        }

        let first_group_map = group_texture.clone();
        let first_groups = groups.clone();
        let mut first_sources = Array1::<i32>::zeros(first_groups.len() * 9 * 4);
        let mut first_scaling_factors = Array1::<f32>::zeros(first_groups.len() * 9 * 4);
        let mut first_neighborhoods = vec![];

        let mut neighborhoods = vec![];

        let merged_vertices = merge_vertices(vertices_world_positions_arr, wp_indices_arr);
        let triangle_edge_neighborhoods = create_triangle_edge_neighborhoods(
            vertices_world_positions_arr,
            wp_indices_arr,
            &merged_vertices,
        );

        let result = PyDict::new(py);

        let sources_nearest_layers = PyList::empty(py);
        let sources_linear_layers = PyList::empty(py);
        let scaling_factors_layers = PyList::empty(py);
        let pooling_spans_layers = PyList::empty(py);
        let pooling_indices_layers = PyList::empty(py);

        let group_textures_layers = PyList::empty(py);

        let world_position_texture_py =
            PyArray3::<f32>::zeros(py, (resolution.x, resolution.y, 3), false);

        for x in 0..resolution.x {
            for y in 0..resolution.y {
                if let Some(world_position) = world_position_texture[[x, y]] {
                    unsafe {
                        *world_position_texture_py.uget_mut([x, y, 0]) = world_position.x;
                        *world_position_texture_py.uget_mut([x, y, 1]) = world_position.y;
                        *world_position_texture_py.uget_mut([x, y, 2]) = world_position.z;
                    }
                }
            }
        }

        // Now, we can process layers
        for layer in 0..n_layers {
            if layer != 0 {
                texel_size *= 2.0;
            }

            if layer > 0 {
                neighborhoods.push(create_group_neighborhoods(
                    &groups,
                    first_group_map.view(),
                    &first_neighborhoods,
                ));
            }

            let conv_terms_nearest_len = groups.len() * 3 * 3;
            let conv_terms_linear_len = conv_terms_nearest_len * 4;
            let sources_nearest = PyArray1::<i32>::new(py, conv_terms_nearest_len, false);
            let sources_linear = PyArray1::<i32>::new(py, conv_terms_linear_len, false);
            let scaling_factors = PyArray1::<f32>::new(py, conv_terms_linear_len, false);

            let group_texture_py = PyArray2::<i32>::new(py, (resolution.x, resolution.y), false);

            unsafe {
                for i in 0..conv_terms_linear_len {
                    *sources_linear.uget_mut([i]) = -1;
                    *scaling_factors.uget_mut([i]) = 0.0;
                }
                for x in 0..resolution.x {
                    for y in 0..resolution.y {
                        if let Some(group) = group_texture[[x, y]] {
                            *group_texture_py.uget_mut([x, y]) = group as i32;
                        } else {
                            *group_texture_py.uget_mut([x, y]) = -1;
                        }
                    }
                }
            }

            let mesh_layer = MeshLayer { mesh_id, layer };

            let mut successful_samples = Vec::new();

            for (group, triangle) in groups.iter().zip(group_triangles.iter().copied()) {
                let mut local_successful_samples = [[false; 3]; 3];

                let target_group = group.id;

                let normal = (triangles[triangle.0].world_positions[1]
                    - triangles[triangle.0].world_positions[0])
                    .cross(
                        triangles[triangle.0].world_positions[2]
                            - triangles[triangle.0].world_positions[0],
                    )
                    .normalize();

                let tangent = project_vector_onto_plane(
                    (triangle.1.x * triangles[triangle.0].tangents[0]
                        + triangle.1.y * triangles[triangle.0].tangents[1]
                        + triangle.1.z * triangles[triangle.0].tangents[2])
                        .normalize(),
                    normal,
                )
                .normalize(); //triangles[triangle.0].tangent(test[group.id % 3], normal);
                let bitangent = normal.cross(tangent).normalize();

                for kernel_x in 0..3 {
                    for kernel_y in 0..3 {
                        if kernel_x == 1 && kernel_y == 1 {
                            continue;
                        }

                        let weight_id = kernel_x + kernel_y * 3;

                        let dx = (kernel_x as f32) - 1.0;
                        let dy = (kernel_y as f32) - 1.0;
                        let dir = dx * tangent + dy * bitangent;
                        let length = if dx == 0.0 || dy == 0.0 {
                            texel_size
                        } else {
                            std::f32::consts::SQRT_2 * texel_size
                        };

                        if layer == 0 {
                            let (triangle_idx, location_wp, _, success) = trace_surface_line(
                                group.position.map(|n| n as f64),
                                dir.map(|n| n as f64),
                                length as f64,
                                triangle.0 as i32,
                                vertices_world_positions_arr,
                                wp_indices_arr,
                                &merged_vertices,
                                &triangle_edge_neighborhoods,
                            );

                            if !success {
                                continue;
                            }

                            let triangle = &triangles[triangle_idx as usize];

                            // transform location_wp to location_uv
                            let barycentric = if let Some(barycentric) =
                                triangle.barycentric_of_wp(location_wp.map(|n| n as f32))
                            {
                                barycentric
                            } else {
                                continue;
                            };
                            let uv = triangle.uv(barycentric);

                            // linear interpolation of uv
                            let (interpolation_coordinates, interpolation_factors) =
                                discretize_uv_bilinear_interpolation_cv(
                                    uv,
                                    resolution.map(|n| n as f32),
                                );

                            let mut interpolation_factors_sum = 0.0;

                            // store convolution terms
                            for i in 0..4 {
                                let source_group = match group_texture[[
                                    interpolation_coordinates[i].x as usize,
                                    interpolation_coordinates[i].y as usize,
                                ]] {
                                    Some(source_group) => source_group,
                                    None => continue,
                                };

                                local_successful_samples[kernel_x][kernel_y] = true;

                                interpolation_factors_sum += interpolation_factors[i];

                                // 9 -> weights in kernel
                                // 4 -> linear interpolation terms
                                let term_id = target_group * 9 * 4 + weight_id * 4 + i;

                                unsafe {
                                    *sources_linear.uget_mut([term_id]) = source_group as i32;
                                    *scaling_factors.uget_mut([term_id]) = interpolation_factors[i];
                                }
                            }

                            if local_successful_samples[kernel_x][kernel_y] {
                                for i in 0..4 {
                                    let term_id = target_group * 9 * 4 + weight_id * 4 + i;
                                    unsafe {
                                        *scaling_factors.uget_mut([term_id]) /=
                                            interpolation_factors_sum;
                                    }
                                }
                            }
                        } else {
                            let mut surface_samples = vec![];
                            let (_, _, _, success) = trace_surface_line_with_samples(
                                &mut surface_samples,
                                group.position.map(|n| n as f64),
                                dir.map(|n| n as f64),
                                length as f64,
                                length as f64 / 8.0,
                                triangle.0 as i32,
                                vertices_world_positions_arr,
                                wp_indices_arr,
                                &merged_vertices,
                                &triangle_edge_neighborhoods,
                            );

                            if !success {
                                continue;
                            }

                            let mut source_group = None;

                            for (triangle_idx, location_wp) in surface_samples.iter().rev().copied()
                            {
                                let triangle = &triangles[triangle_idx as usize];

                                // transform location_wp to location_uv
                                let barycentric = if let Some(barycentric) =
                                    triangle.barycentric_of_wp(location_wp.map(|n| n as f32))
                                {
                                    barycentric
                                } else {
                                    continue;
                                };
                                let uv = triangle.uv(barycentric);

                                // linear interpolation of uv
                                let (interpolation_coordinates, interpolation_factors) =
                                    discretize_uv_bilinear_interpolation_cv(
                                        uv,
                                        resolution.map(|n| n as f32),
                                    );

                                // store convolution terms
                                //let mut best_neighbor = None;
                                for i in 0..4 {
                                    if interpolation_factors[i] == 0.0 {
                                        continue;
                                    }
                                    let candidate_source_group = match group_texture[[
                                        interpolation_coordinates[i].x as usize,
                                        interpolation_coordinates[i].y as usize,
                                    ]] {
                                        Some(candidate_source_group) => candidate_source_group,
                                        None => continue,
                                    };

                                    if neighborhoods[layer].contains(&(
                                        target_group.min(candidate_source_group),
                                        target_group.max(candidate_source_group),
                                    )) {
                                        if source_group.is_none() {
                                            source_group = Some((
                                                candidate_source_group,
                                                interpolation_factors[i],
                                            ));
                                        } else {
                                            let (_, other_interpolation_factor) =
                                                source_group.unwrap();
                                            if interpolation_factors[i] > other_interpolation_factor
                                            {
                                                source_group = Some((
                                                    candidate_source_group,
                                                    interpolation_factors[i],
                                                ));
                                            }
                                        }
                                    }
                                }
                                if source_group.is_some() {
                                    break;
                                }
                            }

                            if let Some((source_group, _)) = source_group {
                                local_successful_samples[kernel_x][kernel_y] = true;

                                let term_id = target_group * 9 * 4 + weight_id * 4;
                                let scaling_factor = 1.0;

                                unsafe {
                                    *sources_linear.uget_mut([term_id]) = source_group as i32;
                                    *scaling_factors.uget_mut([term_id]) = scaling_factor;
                                }
                            }
                        }

                        if local_successful_samples[kernel_x][kernel_y] {
                            let mut successful_sample = None;
                            for i in 0..4 {
                                let term_id = target_group * 9 * 4 + weight_id * 4 + i;
                                let source = unsafe { *sources_linear.uget_mut([term_id]) };
                                if source >= 0 {
                                    successful_sample = Some(source);
                                    break;
                                }
                            }
                            let successful_sample = successful_sample.unwrap();
                            for i in 0..4 {
                                let term_id = target_group * 9 * 4 + weight_id * 4 + i;
                                let source = unsafe { *sources_linear.uget_mut([term_id]) };
                                if source < 0 {
                                    unsafe {
                                        *sources_linear.uget_mut([term_id]) = successful_sample;
                                        *scaling_factors.uget_mut([term_id]) = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }

                // center
                for i in 0..4 {
                    let weight_id = 4;
                    let term_id = target_group * 9 * 4 + weight_id * 4 + i;

                    unsafe {
                        *sources_linear.uget_mut([term_id]) = target_group as i32;
                        if i == 0 {
                            *scaling_factors.uget_mut([term_id]) = 1.0;
                        } else {
                            *scaling_factors.uget_mut([term_id]) = 0.0;
                        }
                    }

                    local_successful_samples[1][1] = true;
                }

                successful_samples.extend_from_slice(&[
                    false, false, false, false, false, false, false, false, false,
                ]);

                for kernel_x in 0..3 {
                    for kernel_y in 0..3 {
                        let weight_id = kernel_x + kernel_y * 3;

                        successful_samples[group.id * 9 + weight_id] =
                            local_successful_samples[kernel_x][kernel_y];

                        if local_successful_samples[kernel_x][kernel_y] {
                            continue;
                        }

                        let mut replacement_kernel_x = 2 - kernel_x;
                        let mut replacement_kernel_y = 2 - kernel_y;

                        if !local_successful_samples[replacement_kernel_x][replacement_kernel_y] {
                            replacement_kernel_x = 1;
                            replacement_kernel_y = 1;
                        }

                        let replacement_weight_id = replacement_kernel_x + replacement_kernel_y * 3;

                        for i in 0..4 {
                            let term_id = target_group * 9 * 4 + weight_id * 4 + i;

                            let replacement_term_id =
                                target_group * 9 * 4 + replacement_weight_id * 4 + i;

                            unsafe {
                                let source_group = *sources_linear.uget([replacement_term_id]);
                                let interpolation_factor =
                                    *scaling_factors.uget([replacement_term_id]);

                                *sources_linear.uget_mut([term_id]) = source_group;
                                *scaling_factors.uget_mut([term_id]) = interpolation_factor;
                            }
                        }
                    }
                }
            }

            self.groups.insert(mesh_layer, groups.clone());

            if layer == 0 {
                for i in 0..first_groups.len() * 9 * 4 {
                    first_sources[[i]] = unsafe { *sources_linear.get(i).unwrap() };
                    first_scaling_factors[[i]] = unsafe { *scaling_factors.get(i).unwrap() };
                }
                first_neighborhoods =
                    sources_to_neighborhoods(first_sources.view(), first_scaling_factors.view());

                neighborhoods.push(HashSet::new());
                for group_a in 0..first_neighborhoods.len() {
                    for group_b in first_neighborhoods[group_a]
                        .iter()
                        .map(|(group_b, _)| *group_b as usize)
                    {
                        neighborhoods[0].insert((group_a.min(group_b), group_a.max(group_b)));
                    }
                }
            }

            for idx_nearest in 0..conv_terms_nearest_len {
                let mut best_scaling_factor = std::f32::NEG_INFINITY;
                let mut best_source = -1;
                for j in 0..4 {
                    let idx_linear = idx_nearest * 4 + j;
                    unsafe {
                        let scaling_factor = *scaling_factors.uget([idx_linear]);
                        if scaling_factor > best_scaling_factor {
                            best_scaling_factor = scaling_factor;
                            best_source = *sources_linear.uget([idx_linear]);
                        }
                    }
                }
                unsafe {
                    *sources_nearest.uget_mut([idx_nearest]) = best_source;
                }
            }

            sources_nearest_layers.append(sources_nearest).unwrap();
            sources_linear_layers.append(sources_linear).unwrap();
            scaling_factors_layers.append(scaling_factors).unwrap();
            group_textures_layers.append(group_texture_py).unwrap();

            if layer < n_layers - 1 {
                let pools = create_voxel_pools(
                    &groups,
                    texel_size * 2.0,
                    first_group_map.view(),
                    &first_neighborhoods,
                );

                let pooling_spans = PyArray1::<u32>::new(py, pools.len() + 1, false);
                let pooling_indices = PyArray1::<u32>::new(py, groups.len(), false);
                let mut accum = 0;
                let mut next_index = 0;

                for (pool_idx, pool) in pools.iter().enumerate() {
                    unsafe {
                        *pooling_spans.uget_mut([pool_idx]) = accum;
                    }
                    accum += pool.iter().count() as u32;

                    for group_idx in pool.iter().copied() {
                        unsafe {
                            *pooling_indices.uget_mut([next_index]) = group_idx as u32;
                            next_index += 1;
                        }
                    }
                }
                unsafe {
                    *pooling_spans.uget_mut([pools.len()]) = accum;
                }

                pooling_spans_layers.append(pooling_spans).unwrap();
                pooling_indices_layers.append(pooling_indices).unwrap();

                for x in 0..resolution.x {
                    for y in 0..resolution.y {
                        group_texture[[x, y]] = None;
                    }
                }

                let mut new_groups = Vec::with_capacity(pools.len());
                let mut new_group_triangles = Vec::with_capacity(pools.len());

                pools
                    .par_iter()
                    .enumerate()
                    .map(|(new_group_id, pool)| {
                        let mut new_texels: Vec<Point2<usize>> = vec![];
                        let mut pool_first_groups: Vec<&Group<f32>> = vec![];
                        for group in pool.iter().copied() {
                            new_texels.extend(&groups[group as usize].texels);
                            for texel in groups[group as usize].texels.iter() {
                                pool_first_groups.push(
                                    &first_groups[first_group_map[[texel.x, texel.y]].unwrap()],
                                );
                            }
                        }
                        let eccentricities =
                            compute_eccentricities(&pool_first_groups, &first_neighborhoods);
                        let mut best_group = None;
                        let mut best_texel = Point2::new(0, 0);
                        let mut best_eccentricity = std::f32::INFINITY;
                        for (pool_first_group_idx, eccentricity) in
                            eccentricities.iter().copied().enumerate()
                        {
                            let pool_first_group_texel =
                                pool_first_groups[pool_first_group_idx].texels[0];
                            if eccentricity < best_eccentricity {
                                best_group = Some(pool_first_group_idx);
                                best_texel = pool_first_group_texel;
                                best_eccentricity = eccentricity;
                            } else if eccentricity == best_eccentricity {
                                if pool_first_group_texel.x < best_texel.x
                                    || (pool_first_group_texel.x == best_texel.x
                                        && pool_first_group_texel.y < best_texel.y)
                                {
                                    best_group = Some(pool_first_group_idx);
                                    best_texel = pool_first_group_texel;
                                }
                            }
                        }
                        let best_group = pool_first_groups[best_group.unwrap()];
                        let best_position = best_group.position;

                        let best_texel_idx = new_texels
                            .iter()
                            .copied()
                            .position(|t| t == best_texel)
                            .unwrap();
                        new_texels.swap(0, best_texel_idx);

                        let new_group = Group {
                            id: new_group_id,
                            position: best_position,
                            texels: new_texels,
                        };

                        (
                            new_group,
                            triangle_texture[[best_texel.x, best_texel.y]].unwrap(),
                        )
                    })
                    .unzip_into_vecs(&mut new_groups, &mut new_group_triangles);
                for (new_group_id, pool) in pools.iter().enumerate() {
                    for group in pool.iter().copied() {
                        for texel in groups[group as usize].texels.iter() {
                            group_texture[[texel.x, texel.y]] = Some(new_group_id);
                        }
                    }
                }

                groups = new_groups;
                group_triangles = new_group_triangles;
            }
        }

        result
            .set_item("sources_nearest".to_object(py), sources_nearest_layers)
            .unwrap();
        result
            .set_item("sources_linear".to_object(py), sources_linear_layers)
            .unwrap();
        result
            .set_item("scaling_factors".to_object(py), scaling_factors_layers)
            .unwrap();
        result
            .set_item("pooling_spans".to_object(py), pooling_spans_layers)
            .unwrap();
        result
            .set_item("pooling_indices".to_object(py), pooling_indices_layers)
            .unwrap();
        result
            .set_item("group_textures".to_object(py), group_textures_layers)
            .unwrap();

        Ok(result.into())
    }
}

fn merge_vertices(
    vertices_world_positions: ArrayView2<f32>,
    indices: ArrayView2<i32>,
) -> HashMap<Point3<u32>, Vec<i32>> {
    let n_vertices = vertices_world_positions.dim().0;
    let n_triangles = indices.dim().0;

    // bit_vertex -> vector of triangles
    let mut merged_vertices = HashMap::<Point3<u32>, Vec<i32>>::new();
    for i in 0..n_vertices {
        let world_position = Point3::new(
            vertices_world_positions[(i, 0)],
            vertices_world_positions[(i, 1)],
            vertices_world_positions[(i, 2)],
        );

        let bit_vertex = world_position.map(|n| n.to_bits());
        merged_vertices.insert(bit_vertex, vec![]);
    }

    for i in 0..n_triangles {
        let vertices: [Point3<f32>; 3] = std::array::from_fn(|v| {
            Point3::new(
                vertices_world_positions[(indices[(i, v)] as usize, 0)],
                vertices_world_positions[(indices[(i, v)] as usize, 1)],
                vertices_world_positions[(indices[(i, v)] as usize, 2)],
            )
        });

        for vertex in vertices.iter() {
            let bit_vertex = vertex.map(|n| n.to_bits());
            merged_vertices.get_mut(&bit_vertex).unwrap().push(i as i32);
        }
    }

    merged_vertices
}

fn create_triangle_edge_neighborhoods(
    vertices_world_positions: ArrayView2<f32>,
    indices: ArrayView2<i32>,
    merged_vertices: &HashMap<Point3<u32>, Vec<i32>>,
) -> Vec<(i32, i8)> {
    let n_triangles = indices.dim().0;

    // idx = [triangle, edge]
    // .0 = triangle_neighbor
    // .1 = self edge index (0, 1, 2)
    let mut triangle_edge_neighborhood: Vec<(i32, i8)> = vec![(-1, -1); n_triangles * 3];

    for i in 0..n_triangles {
        let idxs: [i32; 3] = std::array::from_fn(|n| indices[(i, n)]);

        for (edge_index, edge) in [[0, 1], [1, 2], [2, 0]].iter().enumerate() {
            let vertex_a = Point3::new(
                vertices_world_positions[(idxs[edge[0]] as usize, 0)],
                vertices_world_positions[(idxs[edge[0]] as usize, 1)],
                vertices_world_positions[(idxs[edge[0]] as usize, 2)],
            );

            let vertex_b = Point3::new(
                vertices_world_positions[(idxs[edge[1]] as usize, 0)],
                vertices_world_positions[(idxs[edge[1]] as usize, 1)],
                vertices_world_positions[(idxs[edge[1]] as usize, 2)],
            );

            let bit_vertex_a = vertex_a.map(|n| n.to_bits());
            let bit_vertex_b = vertex_b.map(|n| n.to_bits());

            let triangles_a = &merged_vertices[&bit_vertex_a];
            let triangles_b = &merged_vertices[&bit_vertex_b];

            let mut shared_triangles = vec![];
            for triangle_a in triangles_a.iter().copied() {
                for triangle_b in triangles_b.iter().copied() {
                    // triangle_b != i as i32 is already checked, but leaving it here for explicitness
                    if triangle_a != i as i32 && triangle_b != i as i32 && triangle_a == triangle_b
                    {
                        shared_triangles.push(triangle_a);
                    }
                }
            }

            if !shared_triangles.is_empty() {
                triangle_edge_neighborhood[i * 3 + edge_index] =
                    (shared_triangles[0], edge_index as i8);
            }
        }
    }

    triangle_edge_neighborhood
}

fn create_voxel_pools(
    groups: &[Group<f32>],
    pool_size: f32,
    first_group_map: ArrayView2<Option<usize>>,
    first_neighborhoods: &[Vec<(i32, f32)>],
) -> Vec<Vec<i32>> {
    let mut voxels = HashMap::<Point3<i32>, Vec<i32>>::new();

    for group in groups.iter() {
        let pos = group.position.map(|p| (p / pool_size) as i32);

        voxels
            .entry(pos)
            .and_modify(|e| e.push(group.id as i32))
            .or_insert_with(|| vec![group.id as i32]);
    }

    let mut pools = vec![];
    for pool in voxels.into_iter().map(|e| e.1) {
        let pool_groups = pool
            .iter()
            .copied()
            .map(|g| &groups[g as usize])
            .collect::<Vec<_>>();
        let connected_pools = divide_pool_into_components(
            pool_groups.as_slice(),
            first_group_map,
            first_neighborhoods,
        );
        for connected_pool in connected_pools.into_iter() {
            pools.push(connected_pool);
        }
    }

    pools
}

fn create_group_neighborhoods(
    groups: &[Group<f32>],
    first_group_map: ArrayView2<Option<usize>>,
    first_neighborhoods: &[Vec<(i32, f32)>],
) -> HashSet<(usize, usize)> {
    let mut neighborhoods = HashSet::new();

    let n_first_groups = first_neighborhoods.len();

    let mut first_group_to_group = vec![std::usize::MAX; n_first_groups];
    for group in groups.iter() {
        for texel in group.texels.iter() {
            let first_group = first_group_map[[texel.x, texel.y]].unwrap();
            first_group_to_group[first_group] = group.id;
        }

        neighborhoods.insert((group.id, group.id));
    }

    for first_group in 0..n_first_groups {
        let group = first_group_to_group[first_group];
        for first_neighbor in first_neighborhoods[first_group]
            .iter()
            .map(|n| n.0 as usize)
        {
            let neighbor = first_group_to_group[first_neighbor];

            neighborhoods.insert((group.min(neighbor), group.max(neighbor)));
        }
    }

    neighborhoods
}

fn divide_pool_into_components(
    pool: &[&Group<f32>],
    first_group_map: ArrayView2<Option<usize>>,
    first_neighborhoods: &[Vec<(i32, f32)>],
) -> Vec<Vec<i32>> {
    let mut group_to_component = HashMap::<i32, usize>::new();
    let mut component_to_groups = HashMap::<usize, Vec<i32>>::new();

    let mut first_groups_to_groups = HashMap::new();
    for group in pool.iter() {
        for texel in group.texels.iter() {
            let first_group = first_group_map[[texel.x, texel.y]];
            assert!(first_group.is_some());
            first_groups_to_groups.insert(first_group.unwrap() as i32, group.id as i32);
        }
    }

    let mut component_idx = 0;
    for group in pool.iter() {
        let mut components_to_merge = HashSet::new();

        let current_component =
            if let Some(current_component) = group_to_component.get(&(group.id as i32)) {
                *current_component
            } else {
                let new_component = component_idx;
                component_idx += 1;
                group_to_component.insert(group.id as i32, new_component);
                component_to_groups.insert(new_component, vec![group.id as i32]);
                new_component
            };

        for texel in group.texels.iter() {
            let first_group = first_group_map[[texel.x, texel.y]].unwrap();
            for (first_neighbor, _) in first_neighborhoods[first_group].iter() {
                if let Some(neighbor) = first_groups_to_groups.get(&first_neighbor) {
                    if let Some(component) = group_to_component.get(neighbor) {
                        if *component != current_component {
                            components_to_merge.insert(*component);
                        }
                    } else {
                        group_to_component.insert(*neighbor, current_component);
                        component_to_groups
                            .get_mut(&current_component)
                            .unwrap()
                            .push(*neighbor);
                    }
                }
            }
        }

        for component in components_to_merge.iter().copied() {
            let other_groups = component_to_groups.remove(&component).unwrap();
            for other_group in other_groups.into_iter() {
                *group_to_component.get_mut(&other_group).unwrap() = current_component;
                component_to_groups
                    .get_mut(&current_component)
                    .unwrap()
                    .push(other_group);
            }
        }
    }

    let divided_pools = component_to_groups
        .into_iter()
        .map(|e| e.1)
        .collect::<Vec<_>>();

    divided_pools
}

fn sources_to_neighborhoods(
    sources: ArrayView1<i32>,
    scaling_factors: ArrayView1<f32>,
) -> Vec<Vec<(i32, f32)>> {
    let groups = sources.dim() / 36;
    let mut neighborhoods = vec![HashMap::new(); groups];

    for i in 0..groups {
        for j in 0..36 {
            let w = j / 4;
            let term_id = i * 36 + j;
            let source = sources[[term_id]];
            let scaling_factor = scaling_factors[[term_id]];

            if source < 0 || scaling_factor <= 0.0 {
                continue;
            }

            let w_distance = if w == 4 {
                0.0
            } else if w == 0 || w == 2 || w == 6 || w == 8 {
                std::f32::consts::SQRT_2
            } else {
                1.0
            };

            neighborhoods[i]
                .entry(source)
                .and_modify(|distance| {
                    if w_distance < *distance {
                        *distance = w_distance;
                    }
                })
                .or_insert(w_distance);
            neighborhoods[source as usize]
                .entry(i as i32)
                .and_modify(|distance| {
                    if w_distance < *distance {
                        *distance = w_distance;
                    }
                })
                .or_insert(w_distance);
        }
    }

    neighborhoods
        .into_iter()
        .map(|neighborhood| neighborhood.into_iter().collect())
        .collect()
}

fn compute_eccentricities(groups: &[&Group<f32>], neighborhoods: &[Vec<(i32, f32)>]) -> Vec<f32> {
    let group_ids_to_idxs = groups
        .iter()
        .enumerate()
        .map(|(idx, g)| (g.id as i32, idx))
        .collect::<HashMap<i32, usize>>();
    let mut distances = vec![std::f32::INFINITY; groups.len() * groups.len()];
    let key = |a: usize, b: usize| a * groups.len() + b;

    for a_idx in 0..groups.len() {
        distances[key(a_idx, a_idx)] = 0.0;
    }

    for a_idx in 0..groups.len() {
        let a_id = groups[a_idx].id as i32;

        for (b_id, distance) in neighborhoods[a_id as usize].iter() {
            let distance = *distance;
            if !group_ids_to_idxs.contains_key(&b_id) {
                continue;
            }
            if let Some(b_idx) = group_ids_to_idxs.get(&b_id) {
                if distance < distances[key(a_idx, *b_idx)] {
                    distances[key(a_idx, *b_idx)] = distance;
                    distances[key(*b_idx, a_idx)] = distance;
                }
            }
        }
    }

    for k in 0..groups.len() {
        for i in 0..groups.len() {
            for j in 0..groups.len() {
                let dist_ik_kj = distances[key(i, k)] + distances[key(k, j)];
                if distances[key(i, j)] > dist_ik_kj {
                    distances[key(i, j)] = dist_ik_kj;
                    distances[key(j, i)] = dist_ik_kj;
                }
            }
        }
    }

    let mut eccentricities = vec![std::f32::NEG_INFINITY; groups.len()];
    for a_idx in 0..groups.len() {
        for b_idx in a_idx..groups.len() {
            let distance = distances[key(a_idx, b_idx)];
            if distance > eccentricities[a_idx] {
                eccentricities[a_idx] = distance;
            }
            if distance > eccentricities[b_idx] {
                eccentricities[b_idx] = distance;
            }
        }
    }

    eccentricities
}
