use cgmath::prelude::*;
use cgmath::{Matrix3, Point2, Point3, Vector2, Vector3};

use crate::float::Float;
use crate::triangle::Triangle;

pub fn project_vector_onto_plane<F: Float>(v: Vector3<F>, n: Vector3<F>) -> Vector3<F> {
    let d = v.dot(n);
    let p = n * d;
    v - p
}

pub fn oriented_plane_normal(
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

pub fn rotation_from_two_vectors(v1: Vector3<f64>, v2: Vector3<f64>) -> Matrix3<f64> {
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

pub fn point_line_distance(p: Point3<f64>, a0: Point3<f64>, a1: Point3<f64>) -> (Point3<f64>, f64) {
    if a0 == a1 {
        return (a0, p.distance(a0));
    }

    let d = (a1 - a0) / a1.distance(a0);
    let v = p - a0;
    let t = v.dot(d);
    let i = a0 + t * d;

    return (i, i.distance(p));
}

pub fn line_line_distance(
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

pub fn compute_texel_size<F: Float>(triangles: &[Triangle<F>], resolution: Vector2<usize>) -> F {
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

pub fn discretize_uv_cv(uv: Point2<f32>, resolution: Vector2<f32>) -> (Point2<i32>, Vector2<f32>) {
    let rv = Vector2::new(uv.x * resolution.x, uv.y * resolution.y);
    let p = Vector2::new(rv.x.floor(), rv.y.floor());
    let p_center = p + Vector2::new(0.5, 0.5);

    (Point2::new(p.x as i32, p.y as i32), rv - p_center)
}

pub fn discretize_uv_bilinear_interpolation_inner_cv(
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

pub fn discretize_uv_bilinear_interpolation_cv(
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
