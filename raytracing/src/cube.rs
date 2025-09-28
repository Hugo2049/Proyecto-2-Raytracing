use crate::material::Material;
use crate::ray_intersect::{Intersect, RayIntersect};
use raylib::prelude::*;

#[derive(Clone)]
pub struct Cube {
    pub center: Vector3,
    pub size: f32,
    pub material: Material,
    pub texture: Option<Image>,
}

impl Cube {
    pub fn new(center: Vector3, size: f32, material: Material) -> Self {
        Self {
            center,
            size,
            material,
            texture: None,
        }
    }

    pub fn with_texture(center: Vector3, size: f32, material: Material, texture: Image) -> Self {
        Self {
            center,
            size,
            material,
            texture: Some(texture),
        }
    }

    /// Proper UV calculation for each face
    fn calculate_uv(&self, point: Vector3, normal: Vector3) -> (f32, f32) {
        let local_point = point - self.center;
        let half_size = self.size / 2.0;
        
        let (u, v) = if normal.x.abs() > 0.9 {
            // X faces (left/right walls)
            if normal.x > 0.0 {
                ((-local_point.z + half_size) / self.size, (local_point.y + half_size) / self.size)
            } else {
                ((local_point.z + half_size) / self.size, (local_point.y + half_size) / self.size)
            }
        } else if normal.y.abs() > 0.9 {
            // Y faces (floor/ceiling)
            if normal.y > 0.0 {
                ((local_point.x + half_size) / self.size, (-local_point.z + half_size) / self.size)
            } else {
                ((local_point.x + half_size) / self.size, (local_point.z + half_size) / self.size)
            }
        } else {
            // Z faces (front/back walls)
            if normal.z > 0.0 {
                ((local_point.x + half_size) / self.size, (local_point.y + half_size) / self.size)
            } else {
                ((-local_point.x + half_size) / self.size, (local_point.y + half_size) / self.size)
            }
        };
        
        (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
    }

    /// High quality texture sampling
    fn sample_texture(&mut self, u: f32, v: f32) -> Vector3 {
        if let Some(ref mut texture) = self.texture {
            let u = u.clamp(0.0, 1.0);
            let v = v.clamp(0.0, 1.0);
            
            let x = ((u * (texture.width - 1) as f32).round() as i32).clamp(0, texture.width - 1);
            let y = ((v * (texture.height - 1) as f32).round() as i32).clamp(0, texture.height - 1);
            
            let color = texture.get_color(x, y);
            
            Vector3::new(
                color.r as f32 / 255.0,
                color.g as f32 / 255.0,
                color.b as f32 / 255.0,
            )
        } else {
            Vector3::new(1.0, 1.0, 1.0)
        }
    }

    /// Standard AABB ray intersection - no shortcuts
    fn ray_aabb_intersect(&self, ray_origin: &Vector3, ray_direction: &Vector3) -> Option<(f32, Vector3)> {
        let half_size = self.size * 0.5;
        let min_bounds = self.center - Vector3::new(half_size, half_size, half_size);
        let max_bounds = self.center + Vector3::new(half_size, half_size, half_size);
        
        let inv_dir = Vector3::new(
            if ray_direction.x.abs() < 1e-8 { 
                if ray_direction.x >= 0.0 { 1e8 } else { -1e8 } 
            } else { 
                1.0 / ray_direction.x 
            },
            if ray_direction.y.abs() < 1e-8 { 
                if ray_direction.y >= 0.0 { 1e8 } else { -1e8 } 
            } else { 
                1.0 / ray_direction.y 
            },
            if ray_direction.z.abs() < 1e-8 { 
                if ray_direction.z >= 0.0 { 1e8 } else { -1e8 } 
            } else { 
                1.0 / ray_direction.z 
            }
        );
        
        let t1 = (min_bounds.x - ray_origin.x) * inv_dir.x;
        let t2 = (max_bounds.x - ray_origin.x) * inv_dir.x;
        let t3 = (min_bounds.y - ray_origin.y) * inv_dir.y;
        let t4 = (max_bounds.y - ray_origin.y) * inv_dir.y;
        let t5 = (min_bounds.z - ray_origin.z) * inv_dir.z;
        let t6 = (max_bounds.z - ray_origin.z) * inv_dir.z;
        
        let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));
        
        if tmax < 0.0 || tmin > tmax {
            return None;
        }
        
        let t = if tmin > 0.0 { tmin } else { tmax };
        if t <= 0.0 {
            return None;
        }
        
        let point = *ray_origin + *ray_direction * t;
        let local_point = point - self.center;
        
        // Determine which face was hit
        let abs_local = Vector3::new(local_point.x.abs(), local_point.y.abs(), local_point.z.abs());
        let normal = if abs_local.x >= abs_local.y && abs_local.x >= abs_local.z {
            Vector3::new(local_point.x.signum(), 0.0, 0.0)
        } else if abs_local.y >= abs_local.z {
            Vector3::new(0.0, local_point.y.signum(), 0.0)
        } else {
            Vector3::new(0.0, 0.0, local_point.z.signum())
        };
        
        Some((t, normal))
    }
}

impl RayIntersect for Cube {
    fn ray_intersect(&mut self, ray_origin: &Vector3, ray_direction: &Vector3) -> Intersect {
        if let Some((distance, normal)) = self.ray_aabb_intersect(ray_origin, ray_direction) {
            let point = *ray_origin + *ray_direction * distance;
            
            let (u, v) = self.calculate_uv(point, normal);
            let texture_color = self.sample_texture(u, v);
            
            let mut textured_material = self.material;
            textured_material.diffuse = Vector3::new(
                textured_material.diffuse.x * texture_color.x,
                textured_material.diffuse.y * texture_color.y,
                textured_material.diffuse.z * texture_color.z,
            );
            
            Intersect::new(point, normal, distance, textured_material)
        } else {
            Intersect::empty()
        }
    }
}