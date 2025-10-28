use cgmath::prelude::*;
use cgmath::{dot, Point2, Point3, Vector3};

use crate::float::Float;
use crate::line::Line;
use crate::utils::project_vector_onto_plane;

#[derive(Copy, Clone, Debug)]
pub struct Triangle<F: Float> {
    pub id: usize,
    pub world_positions: [Point3<F>; 3],
    pub normals: [Vector3<F>; 3],
    pub tangents: [Vector3<F>; 3],
    pub uvs: [Point2<F>; 3],
}

impl<F: Float> Triangle<F> {
    pub fn from_wps(a: Point3<F>, b: Point3<F>, c: Point3<F>) -> Triangle<F> {
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

    pub fn barycentric_is_inside(barycentric: Vector3<F>) -> bool {
        let eps = F::from(0.00001).unwrap();
        barycentric[0] >= F::zero() - eps
            && barycentric[0] <= F::one() + eps
            && barycentric[1] >= F::zero() - eps
            && barycentric[1] <= F::one() + eps
            && barycentric[2] >= F::zero() - eps
            && barycentric[2] <= F::one() + eps
            && (barycentric[0] + barycentric[1] + barycentric[2] - F::one()).abs() < eps
    }

    pub fn barycentric_snap_inside(barycentric: Vector3<F>) -> Vector3<F> {
        let mut result = Vector3::new(F::zero(), F::zero(), F::zero());
        result.x = barycentric.x.max(F::zero());
        result.y = barycentric.y.max(F::zero());

        if result.x + result.y > F::one() {
            result.y = (F::one() - result.x).max(F::zero());
        }

        result.z = (F::one() - result.x - result.y).max(F::zero());

        result
    }

    pub fn barycentric_of_uv(&self, uv: Point2<F>) -> Option<Vector3<F>> {
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

    pub fn barycentric_of_wp(&self, wp: Point3<F>) -> Option<Vector3<F>> {
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

    pub fn normal_flat(&self) -> Vector3<F> {
        let e1 = self.world_positions[1] - self.world_positions[0];
        let e2 = self.world_positions[2] - self.world_positions[0];
        e1.cross(e2).normalize()
    }

    pub fn world_position(&self, barycentric: Vector3<F>) -> Point3<F> {
        Point3::from_vec(
            self.world_positions[0].to_vec() * barycentric[0]
                + self.world_positions[1].to_vec() * barycentric[1]
                + self.world_positions[2].to_vec() * barycentric[2],
        )
    }

    pub fn normal(&self, barycentric: Vector3<F>) -> Vector3<F> {
        let normal = self.normals[0] * barycentric[0]
            + self.normals[1] * barycentric[1]
            + self.normals[2] * barycentric[2];
        normal.normalize()
    }

    pub fn tangent(&self, barycentric: Vector3<F>, normal: Vector3<F>) -> Vector3<F> {
        let interpolated_tangent = self.tangents[0] * barycentric[0]
            + self.tangents[1] * barycentric[1]
            + self.tangents[2] * barycentric[2];

        let projected_tangent = project_vector_onto_plane(interpolated_tangent, normal);

        projected_tangent.normalize()
    }

    pub fn uv(&self, barycentric: Vector3<F>) -> Point2<F> {
        Point2::from_vec(
            self.uvs[0].to_vec() * barycentric[0]
                + self.uvs[1].to_vec() * barycentric[1]
                + self.uvs[2].to_vec() * barycentric[2],
        )
    }

    pub fn closest_uv_point(&self, uv: Point2<F>) -> Option<(Point2<F>, F)> {
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
