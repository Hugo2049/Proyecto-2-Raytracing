use raylib::prelude::*;
use std::f32::consts::PI;

mod framebuffer;
mod ray_intersect;
mod cube;
mod camera;
mod light;
mod material;

use framebuffer::Framebuffer;
use ray_intersect::{Intersect, RayIntersect};
use cube::Cube;
use camera::Camera;
use light::Light;
use material::{Material, vector3_to_color};

const ORIGIN_BIAS: f32 = 1e-4;

fn procedural_sky(dir: Vector3) -> Vector3 {
    let d = dir.normalized();
    let t = (d.y + 1.0) * 0.5;

    let green = Vector3::new(0.1, 0.6, 0.2);
    let white = Vector3::new(1.0, 1.0, 1.0);
    let blue = Vector3::new(0.3, 0.5, 1.0);

    if t < 0.54 {
        let k = t / 0.55;
        green * (1.0 - k) + white * k
    } else if t < 0.55 {
        white
    } else if t < 0.8 {
        let k = (t - 0.55) / 0.25;
        white * (1.0 - k) + blue * k
    } else {
        blue
    }
}

#[inline]
fn offset_origin(intersect: &Intersect, direction: &Vector3) -> Vector3 {
    let offset = intersect.normal * ORIGIN_BIAS;
    if direction.dot(intersect.normal) < 0.0 {
        intersect.point - offset
    } else {
        intersect.point + offset
    }
}

#[inline]
fn reflect(incident: &Vector3, normal: &Vector3) -> Vector3 {
    *incident - *normal * 2.0 * incident.dot(*normal)
}

// Simplified shadow casting - early exit optimization
fn cast_shadow_simple(
    intersect: &Intersect,
    light: &Light,
    objects: &mut [Cube],
) -> f32 {
    let light_dir = (light.position - intersect.point).normalized();
    let light_distance = (light.position - intersect.point).length();
    let shadow_ray_origin = offset_origin(intersect, &light_dir);

    // Early exit - check closest objects first
    for object in objects.iter_mut().take(20) { // Limit shadow ray tests
        let shadow_intersect = object.ray_intersect(&shadow_ray_origin, &light_dir);
        if shadow_intersect.is_intersecting && shadow_intersect.distance < light_distance {
            return 0.8; // Softer shadows for better performance
        }
    }
    0.0
}

// Simplified ray casting with aggressive optimizations
pub fn cast_ray_fast(
    ray_origin: &Vector3,
    ray_direction: &Vector3,
    objects: &mut [Cube],
    light: &Light,
    depth: u32,
) -> Vector3 {
    if depth > 1 { // Maximum 1 reflection
        return procedural_sky(*ray_direction);
    }

    let mut intersect = Intersect::empty();
    let mut zbuffer = f32::INFINITY;

    // Find closest intersection
    for object in objects.iter_mut() {
        let i = object.ray_intersect(ray_origin, ray_direction);
        if i.is_intersecting && i.distance < zbuffer {
            zbuffer = i.distance;
            intersect = i;
        }
    }

    if !intersect.is_intersecting {
        return procedural_sky(*ray_direction);
    }

    // Simplified lighting calculation
    let light_dir = (light.position - intersect.point).normalized();
    let diffuse_intensity = intersect.normal.dot(light_dir).max(0.0);
    
    // Skip expensive shadow calculation for distant objects
    let shadow_intensity = if intersect.distance < 10.0 {
        cast_shadow_simple(&intersect, light, objects)
    } else {
        0.0
    };
    
    let light_intensity = light.intensity * (1.0 - shadow_intensity);
    let diffuse = intersect.material.diffuse * (diffuse_intensity * light_intensity);

    // Simplified specular (no expensive pow calculation)
    let view_dir = (*ray_origin - intersect.point).normalized();
    let reflect_dir = reflect(&-light_dir, &intersect.normal).normalized();
    let specular_dot = view_dir.dot(reflect_dir).max(0.0);
    let specular_intensity = if specular_dot > 0.9 { specular_dot } else { 0.0 }; // Sharp cutoff
    
    let light_color_v3 = Vector3::new(
        light.color.r as f32 / 255.0, 
        light.color.g as f32 / 255.0, 
        light.color.b as f32 / 255.0
    );
    let specular = light_color_v3 * (specular_intensity * light_intensity);

    let albedo = intersect.material.albedo;
    diffuse * albedo[0] + specular * albedo[1]
}

// Adaptive quality rendering - lower quality for moving camera
pub fn render_adaptive(
    framebuffer: &mut Framebuffer, 
    objects: &mut [Cube], 
    camera: &Camera, 
    light: &Light,
    quality_level: u32
) {
    let width = framebuffer.width as f32;
    let height = framebuffer.height as f32;
    let aspect_ratio = width / height;
    let fov = PI / 3.0;
    let perspective_scale = (fov * 0.5).tan();

    let pixel_step = match quality_level {
        0 => 4, // Very low quality - 1/16 pixels
        1 => 2, // Low quality - 1/4 pixels  
        _ => 1, // Full quality
    };

    for y in (0..framebuffer.height).step_by(pixel_step) {
        for x in (0..framebuffer.width).step_by(pixel_step) {
            let screen_x = (2.0 * x as f32) / width - 1.0;
            let screen_y = -(2.0 * y as f32) / height + 1.0;

            let screen_x = screen_x * aspect_ratio * perspective_scale;
            let screen_y = screen_y * perspective_scale;

            let ray_direction = Vector3::new(screen_x, screen_y, -1.0).normalized();
            let rotated_direction = camera.basis_change(&ray_direction);

            let pixel_color_v3 = cast_ray_fast(&camera.eye, &rotated_direction, objects, light, 0);
            let pixel_color = vector3_to_color(pixel_color_v3);

            framebuffer.set_current_color(pixel_color);
            
            // Fill pixel block for lower quality modes
            for dy in 0..pixel_step.min((framebuffer.height - y) as usize) {
                for dx in 0..pixel_step.min((framebuffer.width - x) as usize) {
                    framebuffer.set_pixel(x + dx as u32, y + dy as u32);
                }
            }
        }
    }
}

// Create smaller floor for better performance
fn create_floor_optimized(tierra_texture: Image) -> Vec<Cube> {
    let mut cubes = Vec::new();
    let cube_size = 1.2; // Slightly larger cubes
    let floor_size = 6;  // Even smaller 6x6 grid
    let start_offset = -((floor_size - 1) as f32 * cube_size) / 2.0;
    
    let tierra_material = Material::new(
        Vector3::new(1.0, 1.0, 1.0),
        16.0,
        [0.9, 0.1, 0.0, 0.0], // No reflections/transparency
        1.0,
    );
    
    for x in 0..floor_size {
        for z in 0..floor_size {
            let pos_x = start_offset + x as f32 * cube_size;
            let pos_z = start_offset + z as f32 * cube_size;
            let pos_y = -cube_size / 2.0;
            
            let cube = Cube::with_texture(
                Vector3::new(pos_x, pos_y, pos_z),
                cube_size,
                tierra_material,
                tierra_texture.clone(),
            );
            
            cubes.push(cube);
        }
    }
    
    cubes
}

fn main() {
    let window_width = 640;  // Further reduced resolution
    let window_height = 480;
 
    let (mut window, thread) = raylib::init()
        .size(window_width, window_height)
        .title("Fast Diorama Raytracer")
        .log_level(TraceLogLevel::LOG_WARNING)
        .build();

    let mut framebuffer = Framebuffer::new(window_width as u32, window_height as u32);

    // Load texture
    let texture_paths = [
        "src/assets/Piedra.png",
        "./src/assets/Piedra.png",
        "./assets/Piedra.png"
    ];

    let mut tierra_texture = None;
    for path in &texture_paths {
        match Image::load_image(path) {
            Ok(image) => {
                println!("Successfully loaded Tierra texture from: {}", path);
                tierra_texture = Some(image);
                break;
            }
            Err(e) => {
                println!("Failed to load texture from {}: {:?}", path, e);
            }
        }
    }

    let mut objects = if let Some(texture) = tierra_texture {
        create_floor_optimized(texture)
    } else {
        println!("Error: Could not load Tierra texture!");
        let fallback_material = Material::new(
            Vector3::new(0.6, 0.4, 0.2),
            16.0,
            [0.9, 0.1, 0.0, 0.0],
            1.0,
        );
        vec![Cube::new(Vector3::new(0.0, -0.6, 0.0), 1.2, fallback_material)]
    };

    let mut camera = Camera::new(
        Vector3::new(0.0, 2.5, 5.0),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );

    let light = Light::new(
        Vector3::new(1.5, 3.0, 1.5),
        Color::new(255, 255, 255, 255),
        1.3,
    );

    let movement_speed = 0.4;
    let rotation_speed = 0.1;

    println!("High Performance Controls:");
    println!("- WASD: Move around");
    println!("- Q/E: Move up/down");  
    println!("- Arrow keys: Look around");
    println!("- 1/2/3: Quality levels (1=fast, 3=best)");
    println!("- ESC: Exit");
    println!("Optimizations: 6x6 grid, 640x480, adaptive quality, simplified lighting");

    let mut frame_count = 0;
    let mut last_fps_time = std::time::Instant::now();
    let mut quality_level = 1; // Start with low quality
    let mut last_input_time = std::time::Instant::now();

    while !window.window_should_close() {
        let now = std::time::Instant::now();
        let input_active = false;

        // Camera controls
        let mut moved = false;
        if window.is_key_down(KeyboardKey::KEY_W) {
            camera.move_forward(movement_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_S) {
            camera.move_forward(-movement_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_A) {
            camera.move_right(-movement_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_D) {
            camera.move_right(movement_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_Q) {
            camera.move_up(movement_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_E) {
            camera.move_up(-movement_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_LEFT) {
            camera.rotate(-rotation_speed, 0.0);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_RIGHT) {
            camera.rotate(rotation_speed, 0.0);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_UP) {
            camera.rotate(0.0, rotation_speed);
            moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_DOWN) {
            camera.rotate(0.0, -rotation_speed);
            moved = true;
        }

        // Quality control
        if window.is_key_pressed(KeyboardKey::KEY_ONE) {
            quality_level = 0;
            println!("Quality: Very Low (fastest)");
        }
        if window.is_key_pressed(KeyboardKey::KEY_TWO) {
            quality_level = 1;
            println!("Quality: Low");
        }
        if window.is_key_pressed(KeyboardKey::KEY_THREE) {
            quality_level = 2;
            println!("Quality: High (best)");
        }

        // Adaptive quality: use lower quality when moving
        let render_quality = if moved {
            last_input_time = now;
            0 // Very low quality while moving
        } else if now.duration_since(last_input_time).as_millis() < 100 {
            1 // Low quality briefly after stopping
        } else {
            quality_level // Use selected quality when stationary
        };

        // Render
        framebuffer.clear();
        render_adaptive(&mut framebuffer, &mut objects, &camera, &light, render_quality);
        framebuffer.swap_buffers(&mut window, &thread);

        // FPS counter
        frame_count += 1;
        if last_fps_time.elapsed().as_secs() >= 1 {
            println!("FPS: {} (Quality: {})", frame_count, render_quality);
            frame_count = 0;
            last_fps_time = std::time::Instant::now();
        }
    }
}