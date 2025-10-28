use cgmath::prelude::*;
use cgmath::{dot, Point2, Vector2};

use crate::float::Float;

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
