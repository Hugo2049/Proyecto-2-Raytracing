use raylib::prelude::*;

/// A 3D camera for diorama navigation
pub struct Camera {
    pub eye: Vector3,     // Camera position in world coordinates
    pub center: Vector3,  // Point the camera is looking at
    pub up: Vector3,      // Up direction (initially world up, gets orthonormalized)
    pub forward: Vector3, // Direction camera is facing (computed from eye->center)
    pub right: Vector3,   // Right direction (perpendicular to forward and up)
    pub yaw: f32,         // Horizontal rotation angle
    pub pitch: f32,       // Vertical rotation angle
}

impl Camera {
    /// Creates a new camera and computes its initial orientation
    pub fn new(eye: Vector3, center: Vector3, up: Vector3) -> Self {
        let mut camera = Camera {
            eye,
            center,
            up,
            forward: Vector3::zero(),
            right: Vector3::zero(),
            yaw: 0.0,
            pitch: 0.0,
        };
        
        // Calculate initial yaw and pitch from eye and center
        let direction = (center - eye).normalized();
        camera.yaw = direction.z.atan2(direction.x);
        camera.pitch = direction.y.asin();
        
        camera.update_basis_vectors();
        camera
    }

    /// Recomputes the camera's orthonormal basis vectors from eye, center, and up
    pub fn update_basis_vectors(&mut self) {
        // Calculate forward direction from yaw and pitch
        let cos_pitch = self.pitch.cos();
        self.forward = Vector3::new(
            cos_pitch * self.yaw.cos(),
            self.pitch.sin(),
            cos_pitch * self.yaw.sin(),
        );
        
        // Update center based on forward direction
        self.center = self.eye + self.forward;
        
        // Calculate right direction using cross product
        self.right = self.forward.cross(self.up).normalized();
        
        // Recalculate up to ensure perfect orthogonality
        self.up = self.right.cross(self.forward);
    }

    /// Rotates the camera's view direction
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-1.5, 1.5); // Prevent looking too far up/down
        self.update_basis_vectors();
    }

    /// Moves the camera forward/backward along its forward direction
    pub fn move_forward(&mut self, distance: f32) {
        self.eye = self.eye + self.forward * distance;
        self.update_basis_vectors();
    }

    /// Moves the camera left/right along its right direction
    pub fn move_right(&mut self, distance: f32) {
        self.eye = self.eye + self.right * distance;
        self.update_basis_vectors();
    }

    /// Moves the camera up/down along the world up direction
    pub fn move_up(&mut self, distance: f32) {
        self.eye.y += distance;
        self.update_basis_vectors();
    }

    /// Transforms a vector from camera space to world space using basis vectors
    pub fn basis_change(&self, v: &Vector3) -> Vector3 {
        Vector3::new(
            v.x * self.right.x + v.y * self.up.x - v.z * self.forward.x,
            v.x * self.right.y + v.y * self.up.y - v.z * self.forward.y,
            v.x * self.right.z + v.y * self.up.z - v.z * self.forward.z,
        )
    }
}