use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryInto;

use cgmath::prelude::*;
use cgmath::{dot, Matrix3, Point2, Point3, Vector2, Vector3};

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

pub struct Line<F: Float> {
    pub start_point: Point2<F>,
    pub end_point: Point2<F>,
}

impl<F: Float> Line<F> {
    pub fn new(start_point: Point2<F>, end_point: Point2<F>) -> Self {
        Self {
            start_point,
            end_point,
        }
    }

    pub fn intersects_with_endpoints(&self, line: &Line<F>) -> bool {
        let term_a = Line::counter_clockwise(&self.start_point, &line.start_point, &line.end_point);
        let term_b = Line::counter_clockwise(&self.end_point, &line.start_point, &line.end_point);
        let term_c = Line::counter_clockwise(&self.start_point, &self.end_point, &line.start_point);
        let term_d = Line::counter_clockwise(&self.start_point, &self.end_point, &line.end_point);
        term_a != term_b && term_c != term_d
    }

    pub fn counter_clockwise(
        point_a: &Point2<F>,
        point_b: &Point2<F>,
        point_c: &Point2<F>,
    ) -> bool {
        let term_a = (point_c.y - point_a.y) * (point_b.x - point_a.x);
        let term_b = (point_b.y - point_a.y) * (point_c.x - point_a.x);
        term_a > term_b
    }

    pub fn span(&self) -> Vector2<F> {
        self.end_point - self.start_point
    }

    pub fn direction(&self) -> Vector2<F> {
        self.span().normalize()
    }

    pub fn sign(&self, other_point: &Point2<F>) -> bool {
        (other_point.x - self.end_point.x) * (self.start_point.y - self.end_point.y)
            - (self.start_point.x - self.end_point.x) * (other_point.y - self.end_point.y)
            > F::zero()
    }

    pub fn closest_point(&self, point: &Point2<F>) -> Point2<F> {
        let l2 = (self.start_point - self.end_point).magnitude2();
        if l2 == F::zero() {
            return self.start_point;
        }

        let d1 = point - self.start_point;
        let d2 = self.end_point - self.start_point;

        let t = (dot(d1, d2) / l2).min(F::zero()).max(F::one());

        self.start_point
            + Vector2::new(
                t * (self.end_point.x - self.start_point.x),
                t * (self.end_point.y - self.start_point.y),
            )
    }

    pub fn intersection_with_endpoints(&self, line: &Line<F>) -> Option<Point2<F>> {
        let p1 = self.start_point;
        let p2 = self.end_point;
        let p3 = line.start_point;
        let p4 = line.end_point;
        let denom = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
        if denom == F::zero() {
            return None;
        }
        let ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denom;
        if ua < F::zero() || ua > F::one() {
            return None;
        }
        let ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denom;
        if ub < F::zero() || ub > F::one() {
            return None;
        }
        Some(Point2::new(
            p1.x + ua * (p2.x - p1.x),
            p1.y + ua * (p2.y - p1.y),
        ))
    }
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

#[derive(Copy, Clone, Debug)]
struct Triangle<F: Float> {
    id: usize,
    world_positions: [Point3<F>; 3],
    normals: [Vector3<F>; 3],
    tangents: [Vector3<F>; 3],
    uvs: [Point2<F>; 3],
}

impl<F: Float> Triangle<F> {
    fn from_wps(a: Point3<F>, b: Point3<F>, c: Point3<F>) -> Triangle<F> {
        let z = F::zero();
        Triangle {
            id: std::usize::MAX,
            world_positions: [a, b, c],
            normals: [
                Vector3::new(z, z, z),
                Vector3::new(z, z, z),
                Vector3::new(z, z, z),
            ],
            tangents: [
                Vector3::new(z, z, z),
                Vector3::new(z, z, z),
                Vector3::new(z, z, z),
            ],
            uvs: [Point2::new(z, z), Point2::new(z, z), Point2::new(z, z)],
        }
    }

    fn barycentric_is_inside(barycentric: Vector3<F>) -> bool {
        let eps = F::from(0.00001).unwrap();
        barycentric[0] >= F::zero() - eps
            && barycentric[0] <= F::one() + eps
            && barycentric[1] >= F::zero() - eps
            && barycentric[1] <= F::one() + eps
            && barycentric[2] >= F::zero() - eps
            && barycentric[2] <= F::one() + eps
            && (barycentric[0] + barycentric[1] + barycentric[2] - F::one()).abs() < eps
    }

    fn barycentric_snap_inside(barycentric: Vector3<F>) -> Vector3<F> {
        let mut result = Vector3::new(F::zero(), F::zero(), F::zero());
        result.x = barycentric.x.max(F::zero());
        result.y = barycentric.y.max(F::zero());

        if result.x + result.y > F::one() {
            result.y = (F::one() - result.x).max(F::zero());
        }

        result.z = (F::one() - result.x - result.y).max(F::zero());

        result
    }

    fn barycentric_of_uv(&self, uv: Point2<F>) -> Option<Vector3<F>> {
        let v0 = self.uvs[1] - self.uvs[0];
        let v1 = self.uvs[2] - self.uvs[0];
        let v2 = uv - self.uvs[0];
        let d00 = dot(v0, v0);
        let d01 = dot(v0, v1);
        let d11 = dot(v1, v1);
        let d20 = dot(v2, v0);
        let d21 = dot(v2, v1);
        let denom = d00 * d11 - d01 * d01;
        if denom != F::zero() {
            let mut result = Vector3::new(F::zero(), F::zero(), F::zero());
            result[1] = (d11 * d20 - d01 * d21) / denom;
            result[2] = (d00 * d21 - d01 * d20) / denom;
            result[0] = F::one() - result[1] - result[2];
            Some(result)
        } else {
            None
        }
    }

    fn barycentric_of_wp(&self, wp: Point3<F>) -> Option<Vector3<F>> {
        let v0 = self.world_positions[1] - self.world_positions[0];
        let v1 = self.world_positions[2] - self.world_positions[0];
        let v2 = wp - self.world_positions[0];
        let d00 = dot(v0, v0);
        let d01 = dot(v0, v1);
        let d11 = dot(v1, v1);
        let d20 = dot(v2, v0);
        let d21 = dot(v2, v1);
        let denom = d00 * d11 - d01 * d01;
        if denom != F::zero() {
            let mut result = Vector3::new(F::zero(), F::zero(), F::zero());
            result[1] = (d11 * d20 - d01 * d21) / denom;
            result[2] = (d00 * d21 - d01 * d20) / denom;
            result[0] = F::one() - result[1] - result[2];
            Some(result)
        } else {
            None
        }
    }

    fn normal_flat(&self) -> Vector3<F> {
        let e1 = self.world_positions[1] - self.world_positions[0];
        let e2 = self.world_positions[2] - self.world_positions[0];
        e1.cross(e2).normalize()
    }

    fn world_position(&self, barycentric: Vector3<F>) -> Point3<F> {
        Point3::from_vec(
            self.world_positions[0].to_vec() * barycentric[0]
                + self.world_positions[1].to_vec() * barycentric[1]
                + self.world_positions[2].to_vec() * barycentric[2],
        )
    }

    fn normal(&self, barycentric: Vector3<F>) -> Vector3<F> {
        let normal = self.normals[0] * barycentric[0]
            + self.normals[1] * barycentric[1]
            + self.normals[2] * barycentric[2];
        normal.normalize()
    }

    fn tangent(&self, barycentric: Vector3<F>, normal: Vector3<F>) -> Vector3<F> {
        let interpolated_tangent = self.tangents[0] * barycentric[0]
            + self.tangents[1] * barycentric[1]
            + self.tangents[2] * barycentric[2];

        let projected_tangent = project_vector_onto_plane(interpolated_tangent, normal);

        projected_tangent.normalize()
    }

    fn uv(&self, barycentric: Vector3<F>) -> Point2<F> {
        Point2::from_vec(
            self.uvs[0].to_vec() * barycentric[0]
                + self.uvs[1].to_vec() * barycentric[1]
                + self.uvs[2].to_vec() * barycentric[2],
        )
    }

    fn closest_uv_point(&self, uv: Point2<F>) -> Option<(Point2<F>, F)> {
        if let Some(barycentric) = self.barycentric_of_uv(uv) {
            if Self::barycentric_is_inside(barycentric) {
                return Some((uv, F::zero()));
            }

            let mut result = Point2::new(F::zero(), F::zero());
            let mut min_distance2 = F::infinity();

            for vertex_idxs in [[0, 1], [1, 2], [2, 0]].iter() {
                let a = self.uvs[vertex_idxs[0]];
                let b = self.uvs[vertex_idxs[1]];

                let line = Line::new(a, b);
                let closest_point = line.closest_point(&uv);
                let distance2 = uv.distance2(closest_point);
                if distance2 < min_distance2 {
                    result = closest_point;
                    min_distance2 = distance2;
                }
            }

            Some((result, min_distance2.sqrt()))
        } else {
            None
        }
    }
}

fn compute_texel_size<F: Float>(triangles: &[Triangle<F>], resolution: Vector2<usize>) -> F {
    let resolution = resolution.cast().unwrap();

    let mut pixel_sizes: Vec<F> = triangles
        .iter()
        .filter_map(|triangle| {
            let a_uv_v = triangle.uvs[1] - triangle.uvs[0];
            let b_uv_v = triangle.uvs[2] - triangle.uvs[0];
            let area_uv = a_uv_v
                .extend(F::zero())
                .cross(b_uv_v.extend(F::zero()))
                .magnitude()
                * resolution.x
                * resolution.y;

            let a_ws_v = triangle.world_positions[1] - triangle.world_positions[0];
            let b_ws_v = triangle.world_positions[2] - triangle.world_positions[0];
            let area_ws = a_ws_v.cross(b_ws_v).magnitude();

            let texel_size = (area_ws / area_uv).sqrt();
            if texel_size.is_nan() {
                None
            } else {
                Some(texel_size)
            }
        })
        .collect();

    pixel_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if pixel_sizes.len() % 2 != 0 {
        pixel_sizes[pixel_sizes.len() / 2]
    } else {
        (pixel_sizes[(pixel_sizes.len() - 1) / 2] + pixel_sizes[pixel_sizes.len() / 2])
            / (F::one() + F::one())
    }
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
        let triangle_edge_neighborhoods = create_triangle_edge_neighborhoods(vertices_world_positions_arr, wp_indices_arr, &merged_vertices);

        let result = PyDict::new(py);

        let sources_nearest_layers = PyList::empty(py);
        let sources_linear_layers = PyList::empty(py);
        let scaling_factors_layers = PyList::empty(py);
        let pooling_spans_layers = PyList::empty(py);
        let pooling_indices_layers = PyList::empty(py);

        let group_textures_layers = PyList::empty(py);

        let world_position_texture_py = PyArray3::<f32>::zeros(py, (resolution.x, resolution.y, 3), false);

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

        for layer in 0..n_layers {
            if layer != 0 {
                texel_size *= 2.0;
            }

            if layer > 0 {
                neighborhoods.push(create_group_neighborhoods(&groups, first_group_map.view(), &first_neighborhoods));
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

            let mesh_layer = MeshLayer {
                mesh_id,
                layer
            };

            let mut successful_samples = Vec::new();

            for (group, triangle) in groups.iter().zip(group_triangles.iter().copied()) {
                let mut local_successful_samples = [[false; 3]; 3];

                let target_group = group.id;

                let normal = 
                    (triangles[triangle.0].world_positions[1] - triangles[triangle.0].world_positions[0])
                        .cross(triangles[triangle.0].world_positions[2] - triangles[triangle.0].world_positions[0]).normalize();

                let tangent = project_vector_onto_plane((triangle.1.x * triangles[triangle.0].tangents[0] + triangle.1.y * triangles[triangle.0].tangents[1] + triangle.1.z * triangles[triangle.0].tangents[2]).normalize(), normal).normalize(); //triangles[triangle.0].tangent(test[group.id % 3], normal);
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
                            let (triangle_idx, location_wp, _, success) = trace_surface_line(group.position.map(|n| n as f64), dir.map(|n| n as f64), length as f64, triangle.0 as i32, vertices_world_positions_arr, wp_indices_arr, &merged_vertices, &triangle_edge_neighborhoods);

                            if !success {
                                continue;
                            }

                            let triangle = &triangles[triangle_idx as usize];

                            // transform location_wp to location_uv
                            let barycentric = if let Some(barycentric) = triangle.barycentric_of_wp(location_wp.map(|n| n as f32)) {
                                barycentric
                            } else {
                                continue;
                            };
                            let uv = triangle.uv(barycentric);

                            // linear interpolation of uv
                            let (interpolation_coordinates, interpolation_factors) = discretize_uv_bilinear_interpolation_cv(uv, resolution.map(|n| n as f32));

                            let mut interpolation_factors_sum = 0.0;

                            // store convolution terms
                            for i in 0..4 {
                                let source_group = match group_texture[[interpolation_coordinates[i].x as usize, interpolation_coordinates[i].y as usize]] {
                                    Some(source_group) => source_group,
                                    None => continue,
                                };

                                local_successful_samples[kernel_x][kernel_y] = true;
                                
                                interpolation_factors_sum += interpolation_factors[i];

                                // 9 -> weights in kernel
                                // 4 -> linear interpolation terms
                                let term_id = target_group * 9 * 4 
                                    + weight_id * 4
                                    + i;

                                unsafe {
                                    *sources_linear.uget_mut([term_id]) = source_group as i32;
                                    *scaling_factors.uget_mut([term_id]) = interpolation_factors[i];
                                }
                            }

                            if local_successful_samples[kernel_x][kernel_y] {
                                for i in 0..4 {
                                    let term_id = target_group * 9 * 4 
                                        + weight_id * 4
                                        + i;
                                    unsafe {
                                        *scaling_factors.uget_mut([term_id]) /= interpolation_factors_sum;
                                    }
                                }
                            }
                        } else {
                            let mut surface_samples = vec![];
                            let (_, _, _, success) = trace_surface_line_with_samples(&mut surface_samples, group.position.map(|n| n as f64), dir.map(|n| n as f64), length as f64, length as f64 / 8.0, triangle.0 as i32, vertices_world_positions_arr, wp_indices_arr, &merged_vertices, &triangle_edge_neighborhoods);

                            if !success {
                                continue;
                            }

                            let mut source_group = None;

                            for (triangle_idx, location_wp) in surface_samples.iter().rev().copied() {
                                let triangle = &triangles[triangle_idx as usize];

                                // transform location_wp to location_uv
                                let barycentric = if let Some(barycentric) = triangle.barycentric_of_wp(location_wp.map(|n| n as f32)) {
                                    barycentric
                                } else {
                                    continue;
                                };
                                let uv = triangle.uv(barycentric);

                                // linear interpolation of uv
                                let (interpolation_coordinates, interpolation_factors) = discretize_uv_bilinear_interpolation_cv(uv, resolution.map(|n| n as f32));

                                // store convolution terms
                                //let mut best_neighbor = None;
                                for i in 0..4 {
                                    if interpolation_factors[i] == 0.0 {
                                        continue;
                                    }
                                    let candidate_source_group = match group_texture[[interpolation_coordinates[i].x as usize, interpolation_coordinates[i].y as usize]] {
                                        Some(candidate_source_group) => candidate_source_group,
                                        None => continue,
                                    };

                                    if neighborhoods[layer].contains(&(target_group.min(candidate_source_group), target_group.max(candidate_source_group))) {
                                        if source_group.is_none() {
                                            source_group = Some((candidate_source_group, interpolation_factors[i]));
                                        } else {
                                            let (_, other_interpolation_factor) = source_group.unwrap();
                                            if interpolation_factors[i] > other_interpolation_factor {
                                                source_group = Some((candidate_source_group, interpolation_factors[i]));
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

                                let term_id = target_group * 9 * 4 
                                        + weight_id * 4;
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
                                let term_id = target_group * 9 * 4 
                                    + weight_id * 4
                                    + i;
                                let source = unsafe { *sources_linear.uget_mut([term_id]) };
                                if source >= 0 {
                                    successful_sample = Some(source);
                                    break;
                                }
                            }
                            let successful_sample = successful_sample.unwrap();
                            for i in 0..4 {
                                let term_id = target_group * 9 * 4 
                                    + weight_id * 4
                                    + i;
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
                    let term_id = target_group * 9 * 4 
                        + weight_id * 4
                        + i;
                    
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

                successful_samples.extend_from_slice(&[false, false, false, false, false, false, false, false, false]);

                for kernel_x in 0..3 {
                    for kernel_y in 0..3 {
                        let weight_id = kernel_x + kernel_y * 3;

                        successful_samples[group.id * 9 + weight_id] = local_successful_samples[kernel_x][kernel_y];

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
                            let term_id = target_group * 9 * 4 
                                + weight_id * 4
                                + i;
                            
                            let replacement_term_id = target_group * 9 * 4 
                                + replacement_weight_id * 4
                                + i;
            
                            unsafe {
                                let source_group = *sources_linear.uget([replacement_term_id]);
                                let interpolation_factor = *scaling_factors.uget([replacement_term_id]);
                
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
                first_neighborhoods = sources_to_neighborhoods(first_sources.view(), first_scaling_factors.view());

                neighborhoods.push(HashSet::new());
                for group_a in 0..first_neighborhoods.len() {
                    for group_b in first_neighborhoods[group_a].iter().map(|(group_b, _)| *group_b as usize) {
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
                let pools = create_voxel_pools(&groups, texel_size * 2.0, first_group_map.view(), &first_neighborhoods);

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
                                pool_first_groups.push(&first_groups[first_group_map[[texel.x, texel.y]].unwrap()]);
                            }
                        }
                        let eccentricities = compute_eccentricities(&pool_first_groups, &first_neighborhoods);
                        let mut best_group = None;
                        let mut best_texel = Point2::new(0, 0);
                        let mut best_eccentricity = std::f32::INFINITY;
                        for (pool_first_group_idx, eccentricity) in eccentricities.iter().copied().enumerate() {
                            let pool_first_group_texel = pool_first_groups[pool_first_group_idx].texels[0];
                            if eccentricity < best_eccentricity {
                                best_group = Some(pool_first_group_idx);
                                best_texel = pool_first_group_texel;
                                best_eccentricity = eccentricity;
                            } else if eccentricity == best_eccentricity {
                                if pool_first_group_texel.x < best_texel.x || (pool_first_group_texel.x == best_texel.x && pool_first_group_texel.y < best_texel.y) {
                                    best_group = Some(pool_first_group_idx);
                                    best_texel = pool_first_group_texel;
                                }
                            }
                        }
                        let best_group = pool_first_groups[best_group.unwrap()];
                        let best_position = best_group.position;

                        let best_texel_idx = new_texels.iter().copied().position(|t| t == best_texel).unwrap();
                        new_texels.swap(0, best_texel_idx);

                        let new_group = Group {
                            id: new_group_id,
                            position: best_position,
                            texels: new_texels,
                        };

                        (new_group, triangle_texture[[best_texel.x, best_texel.y]].unwrap())
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

        result.set_item("sources_nearest".to_object(py), sources_nearest_layers).unwrap();
        result.set_item("sources_linear".to_object(py), sources_linear_layers).unwrap();
        result.set_item("scaling_factors".to_object(py), scaling_factors_layers).unwrap();
        result.set_item("pooling_spans".to_object(py), pooling_spans_layers).unwrap();
        result.set_item("pooling_indices".to_object(py), pooling_indices_layers).unwrap();
        result.set_item("group_textures".to_object(py), group_textures_layers).unwrap();

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

fn trace_surface_line_with_samples(
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

fn trace_surface_line(
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

fn compute_discontinuity_groups(
    group_tangent_discontinuity_factors: &[f32],
    neighborhoods: &[Vec<(i32, f32)>],
    tangent_discontinuity_threshold: f32,
    tangent_discontinuity_size: usize,
) -> Vec<i32> {
    let mut to_explore = VecDeque::new();
    let mut discovered = HashSet::new();

    for (group_id, discontinuity_factor) in group_tangent_discontinuity_factors
        .iter()
        .copied()
        .enumerate()
    {
        if discontinuity_factor < tangent_discontinuity_threshold {
            to_explore.push_back((group_id as i32, 0));
            discovered.insert(group_id as i32);
        }
    }

    while let Some((group_id, distance)) = to_explore.pop_front() {
        if distance > tangent_discontinuity_size {
            continue;
        }

        for (neighbor, _) in neighborhoods[group_id as usize].iter().copied() {
            if !discovered.contains(&neighbor) {
                to_explore.push_back((neighbor, distance + 1));
                discovered.insert(neighbor);
            }
        }
    }

    discovered.into_iter().collect()
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

fn point_line_distance(p: Point3<f64>, a0: Point3<f64>, a1: Point3<f64>) -> (Point3<f64>, f64) {
    if a0 == a1 {
        return (a0, p.distance(a0));
    }

    let d = (a1 - a0) / a1.distance(a0);
    let v = p - a0;
    let t = v.dot(d);
    let i = a0 + t * d;

    return (i, i.distance(p));
}

fn line_line_distance(
    a0: Point3<f64>,
    a1: Point3<f64>,
    b0: Point3<f64>,
    b1: Point3<f64>,
    clamp_a0: bool,
    clamp_a1: bool,
    clamp_b0: bool,
    clamp_b1: bool,
) -> (Point3<f64>, Point3<f64>, f64) {
    let a = a1 - a0;
    let b = b1 - b0;
    let a_magnitude = a.magnitude();
    let b_magnitude = b.magnitude();

    let a_normalized = a / a_magnitude;
    let b_normalized = b / b_magnitude;

    let cross = a_normalized.cross(b_normalized);
    let denom = cross.magnitude2();

    if denom.abs() < 0.00001 {
        let d0 = a_normalized.dot(b0 - a0);

        if clamp_a0 || clamp_a1 || clamp_b0 || clamp_b1 {
            let d1 = a_normalized.dot(b1 - a0);

            if d0 <= 0.0 && 0.0 >= d1 {
                if clamp_a0 && clamp_b1 {
                    if d0.abs() < d1.abs() {
                        return (a0, b1, (a0 - b0).magnitude());
                    }
                    return (a0, b1, (a0 - b1).magnitude());
                }
            } else if d0 >= a_magnitude && a_magnitude <= d1 {
                if clamp_a1 || clamp_b0 {
                    if d0.abs() < d1.abs() {
                        return (a1, b0, (a1 - b0).magnitude());
                    }
                    return (a1, b1, (a1 - b1).magnitude());
                }
            }
        }

        return (
            Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            (((d0 * a_normalized) + a0.to_vec()) - b0.to_vec()).magnitude(),
        );
    }

    let t = b0 - a0;
    let det_a = Matrix3::from_cols(t, b_normalized, cross).determinant();
    let det_b = Matrix3::from_cols(t, a_normalized, cross).determinant();

    let t0 = det_a / denom;
    let t1 = det_b / denom;

    let mut ap = a0 + (a_normalized * t0);
    let mut bp = b0 + (b_normalized * t1);

    if clamp_a0 || clamp_a1 || clamp_b0 || clamp_b1 {
        if clamp_a0 && t0 < 0.0 {
            ap = a0;
        } else if clamp_a1 && t0 > a_magnitude {
            ap = a1;
        }

        if clamp_b0 && t1 < 0.0 {
            bp = b0;
        } else if clamp_b1 && t1 > b_magnitude {
            bp = b1;
        }

        if (clamp_a0 && t0 < 0.0) || (clamp_a1 && t0 > a_magnitude) {
            let mut dot = b_normalized.dot(ap - b0);
            if clamp_b0 && dot < 0.0 {
                dot = 0.0;
            } else if clamp_b1 && dot > b_magnitude {
                dot = b_magnitude;
            }
            bp = b0 + (b_normalized * dot);
        }

        if (clamp_b0 && t1 < 0.0) || (clamp_b1 && t1 > b_magnitude) {
            let mut dot = a_normalized.dot(bp - a0);
            if clamp_a0 && dot < 0.0 {
                dot = 0.0;
            } else if clamp_a1 && dot > a_magnitude {
                dot = a_magnitude;
            }
            ap = a0 + (a_normalized * dot);
        }
    }

    (ap, bp, (ap - bp).magnitude())
}

fn oriented_plane_normal(
    source: Point3<f64>,
    point_a: Point3<f64>,
    point_b: Point3<f64>,
    v: Vector3<f64>,
) -> Vector3<f64> {
    let l = point_b - point_a;
    let m = point_a + l * 0.5;
    let c = l.cross(v).normalize();
    let o = source - m;
    if o.dot(c) >= 0.0 {
        c
    } else {
        -c
    }
}

fn project_vector_onto_plane<F: Float>(v: Vector3<F>, n: Vector3<F>) -> Vector3<F> {
    let d = v.dot(n);
    let p = n * d;
    v - p
}

fn rotation_from_two_vectors(v1: Vector3<f64>, v2: Vector3<f64>) -> Matrix3<f64> {
    let mut axis = v1.cross(v2);
    let mut angle_sin = axis.magnitude();
    let mut angle_cos = v1.dot(v2);

    axis /= angle_sin;

    if angle_sin < 0.0001 {
        if angle_cos > 0.0 {
            return Matrix3::one();
        } else {
            axis = if v1[0] > v1[1] && v1[0] > v1[2] {
                Vector3::new(-v1[1] - v1[2], v1[0], v1[0])
            } else if v1[1] > v1[2] {
                Vector3::new(v1[1], -v1[0] - v1[2], v1[1])
            } else {
                Vector3::new(v1[2], v1[2], -v1[0] - v1[1])
            };

            axis = axis.normalize();
            angle_sin = 0.0;
            angle_cos = -1.0;
        }
    }

    let ico = 1.0 - angle_cos;
    let nsi_0 = axis[0] * angle_sin;
    let nsi_1 = axis[1] * angle_sin;
    let nsi_2 = axis[2] * angle_sin;

    let n_00 = (axis[0] * axis[0]) * ico;
    let n_01 = (axis[0] * axis[1]) * ico;
    let n_11 = (axis[1] * axis[1]) * ico;
    let n_02 = (axis[0] * axis[2]) * ico;
    let n_12 = (axis[1] * axis[2]) * ico;
    let n_22 = (axis[2] * axis[2]) * ico;

    let mut mat = Matrix3::zero();
    mat[0][0] = n_00 + angle_cos;
    mat[0][1] = n_01 + nsi_2;
    mat[0][2] = n_02 - nsi_1;
    mat[1][0] = n_01 - nsi_2;
    mat[1][1] = n_11 + angle_cos;
    mat[1][2] = n_12 + nsi_0;
    mat[2][0] = n_02 + nsi_1;
    mat[2][1] = n_12 - nsi_0;
    mat[2][2] = n_22 + angle_cos;

    mat
}

fn discretize_uv_cv(uv: Point2<f32>, resolution: Vector2<f32>) -> (Point2<i32>, Vector2<f32>) {
    let rv = Vector2::new(uv.x * resolution.x, uv.y * resolution.y);
    let p = Vector2::new(rv.x.floor(), rv.y.floor());
    let p_center = p + Vector2::new(0.5, 0.5);

    (Point2::new(p.x as i32, p.y as i32), rv - p_center)
}

fn discretize_uv_bilinear_interpolation_inner_cv(
    sample_coordinates: &mut Point2<i32>,
    sample_factors: &mut Vector2<f32>,
    coordinate: i32,
    factor: f32,
    resolution: f32,
) {
    if factor == 1.0 {
        sample_coordinates.x = coordinate;
        sample_coordinates.y = coordinate;
        sample_factors.x = 1.0;
        sample_factors.y = 0.0;
    } else if factor >= 0.0 {
        sample_coordinates.x = coordinate;
        sample_coordinates.y = (coordinate + 1).min(resolution as i32 - 1);
        sample_factors.x = 1.0 - factor;
        sample_factors.y = factor;
    } else if coordinate == 0 {
        sample_coordinates.x = coordinate;
        sample_coordinates.y = coordinate;
        sample_factors.x = factor.abs();
        sample_factors.y = 1.0 + factor;
    } else {
        sample_coordinates.x = (coordinate - 1).max(0);
        sample_coordinates.y = coordinate;
        sample_factors.x = factor.abs();
        sample_factors.y = 1.0 + factor;
    }
}

fn discretize_uv_bilinear_interpolation_cv(
    uv: Point2<f32>,
    resolution: Vector2<f32>,
) -> ([Point2<i32>; 4], [f32; 4]) {
    let (coordinate, factor) = discretize_uv_cv(uv, resolution);

    let mut xs_coordinates = Point2::new(0, 0);
    let mut xs_factors = Vector2::new(0.0, 0.0);
    discretize_uv_bilinear_interpolation_inner_cv(
        &mut xs_coordinates,
        &mut xs_factors,
        coordinate.x,
        factor.x,
        resolution.x,
    );

    let mut ys_coordinates = Point2::new(0, 0);
    let mut ys_factors = Vector2::new(0.0, 0.0);
    discretize_uv_bilinear_interpolation_inner_cv(
        &mut ys_coordinates,
        &mut ys_factors,
        coordinate.y,
        factor.y,
        resolution.y,
    );

    let mut coordinates = [Point2::new(0, 0); 4];
    let mut factors = [0.0; 4];

    coordinates[0] = Point2::new(xs_coordinates.x, ys_coordinates.x);
    factors[0] = xs_factors.x * ys_factors.x;

    coordinates[1] = Point2::new(xs_coordinates.x, ys_coordinates.y);
    factors[1] = xs_factors.x * ys_factors.y;

    coordinates[2] = Point2::new(xs_coordinates.y, ys_coordinates.x);
    factors[2] = xs_factors.y * ys_factors.x;

    coordinates[3] = Point2::new(xs_coordinates.y, ys_coordinates.y);
    factors[3] = xs_factors.y * ys_factors.y;

    (coordinates, factors)
}
