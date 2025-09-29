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

// Performance settings - adjusted for reflections
const ADAPTIVE_RENDER: bool = true;
const MIN_RENDER_SCALE: f32 = 0.125; // Even lower for moving
const MID_RENDER_SCALE: f32 = 0.5;   // Medium quality
const MAX_RENDER_SCALE: f32 = 0.75;  // Reduced max quality
const MAX_RAY_DEPTH: u32 = 2;        // Enable reflections (was 0)
const FRUSTUM_CULLING: bool = true;
const EARLY_RAY_TERMINATION: bool = false; // Disabled - causing holes

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

// Optimized shadow casting - simplified for performance
fn cast_shadow(
    intersect: &Intersect,
    light: &Light,
    objects: &mut [Cube],
) -> f32 {
    let light_dir = (light.position - intersect.point).normalized();
    let light_distance = (light.position - intersect.point).length();
    let shadow_ray_origin = offset_origin(intersect, &light_dir);

    // Early exit for distant lights
    if light_distance > 25.0 {
        return 0.2; // Light shadow for distant surfaces
    }

    // Check all objects for shadows - no early termination to prevent holes
    for object in objects.iter_mut() {
        let shadow_intersect = object.ray_intersect(&shadow_ray_origin, &light_dir);
        if shadow_intersect.is_intersecting && shadow_intersect.distance < light_distance - 0.01 {
            return 0.8; // Reduced shadow intensity
        }
    }
    0.0
}

// Frustum culling - less aggressive to prevent holes
fn is_in_frustum(cube_center: Vector3, _cube_size: f32, camera: &Camera, _fov: f32, _aspect: f32) -> bool {
    if !FRUSTUM_CULLING {
        return true;
    }
    
    let to_cube = cube_center - camera.eye;
    let distance = to_cube.length();
    
    // More conservative distance culling
    if distance > 35.0 {
        return false;
    }
    
    // Very conservative frustum check - only cull objects clearly behind
    let forward_dot = to_cube.normalized().dot(camera.forward);
    if forward_dot < -0.5 {  // Much more lenient
        return false;
    }
    
    true
}

// Enhanced ray casting with reflections and transparency
pub fn cast_ray(
    ray_origin: &Vector3,
    ray_direction: &Vector3,
    objects: &mut [Cube],
    light: &Light,
    depth: u32,
    camera: &Camera,
    fov: f32,
    aspect: f32,
) -> Vector3 {
    if depth > MAX_RAY_DEPTH {
        return procedural_sky(*ray_direction);
    }

    let mut intersect = Intersect::empty();
    let mut zbuffer = f32::INFINITY;

    // Find closest intersection - check all visible objects
    for object in objects.iter_mut() {
        // Only use conservative frustum culling
        if !is_in_frustum(object.center, object.size, camera, fov, aspect) {
            continue;
        }
        
        let i = object.ray_intersect(ray_origin, ray_direction);
        if i.is_intersecting && i.distance < zbuffer {
            zbuffer = i.distance;
            intersect = i;
        }
    }

    if !intersect.is_intersecting {
        return procedural_sky(*ray_direction);
    }

    // Simplified lighting model
    let light_dir = (light.position - intersect.point).normalized();
    let light_distance = (light.position - intersect.point).length();
    
    // Brighter ambient for better visibility
    let ambient = Vector3::new(0.1, 0.1, 0.15);
    
    // Simplified shadow calculation
    let shadow_intensity = if light_distance < 20.0 {
        cast_shadow(&intersect, light, objects)
    } else {
        0.1 // Very light shadow for distant surfaces
    };
    
    let light_visibility = 1.0 - shadow_intensity;
    let distance_falloff = 1.0 / (1.0 + light_distance * light_distance * 0.005);
    
    let diffuse_intensity = intersect.normal.dot(light_dir).max(0.0);
    let light_intensity = light.intensity * light_visibility * distance_falloff;
    
    let diffuse = intersect.material.diffuse * (diffuse_intensity * light_intensity);
    
    // Very simplified specular - only for close surfaces
    let specular = if light_distance < 8.0 && depth == 0 {
        let view_dir = (*ray_origin - intersect.point).normalized();
        let reflect_dir = reflect(&-light_dir, &intersect.normal).normalized();
        let specular_intensity = view_dir.dot(reflect_dir).max(0.0).powf(20.0);
        
        let light_color_v3 = Vector3::new(
            light.color.r as f32 / 255.0, 
            light.color.g as f32 / 255.0, 
            light.color.b as f32 / 255.0
        );
        light_color_v3 * (specular_intensity * light_intensity * 0.2)
    } else {
        Vector3::zero()
    };

    // Reflections for reflective materials (diamonds)
    let mut reflection_color = Vector3::zero();
    if intersect.material.albedo[2] > 0.0 && depth < MAX_RAY_DEPTH {
        let reflect_dir = reflect(ray_direction, &intersect.normal).normalized();
        let reflect_origin = offset_origin(&intersect, &reflect_dir);
        reflection_color = cast_ray(&reflect_origin, &reflect_dir, objects, light, depth + 1, camera, fov, aspect);
    }

    // Refraction/transparency for transparent materials (leaves)
    let mut refract_color = Vector3::zero();
    if intersect.material.albedo[3] > 0.0 && depth < MAX_RAY_DEPTH {
        // Simple transparency - just continue the ray through the object
        let refract_origin = offset_origin(&intersect, ray_direction);
        refract_color = cast_ray(&refract_origin, ray_direction, objects, light, depth + 1, camera, fov, aspect);
    }

    let albedo = intersect.material.albedo;
    let final_color = diffuse * albedo[0] + specular * albedo[1] + reflection_color * albedo[2] + refract_color * albedo[3] + ambient;
    
    Vector3::new(
        final_color.x.min(1.0),
        final_color.y.min(1.0),
        final_color.z.min(1.0)
    )
}

// Fixed adaptive rendering with proper black screen elimination
pub fn render_adaptive(
    framebuffer: &mut Framebuffer, 
    objects: &mut [Cube], 
    camera: &Camera, 
    light: &Light,
    render_scale: f32,
) {
    let width = framebuffer.width;
    let height = framebuffer.height;
    let aspect_ratio = width as f32 / height as f32;
    let fov = PI / 3.0;
    let perspective_scale = (fov * 0.5).tan();

    // Ensure minimum render size and handle edge cases
    let render_width = ((width as f32 * render_scale).round() as u32).max(1).min(width);
    let render_height = ((height as f32 * render_scale).round() as u32).max(1).min(height);

    // If render scale is close to 1.0, just render at full resolution
    if render_scale >= 0.95 {
        // Full resolution rendering
        for y in 0..height {
            for x in 0..width {
                let screen_x = (2.0 * x as f32) / width as f32 - 1.0;
                let screen_y = -(2.0 * y as f32) / height as f32 + 1.0;
                let screen_x = screen_x * aspect_ratio * perspective_scale;
                let screen_y = screen_y * perspective_scale;

                let ray_direction = Vector3::new(screen_x, screen_y, -1.0).normalized();
                let rotated_direction = camera.basis_change(&ray_direction);

                let pixel_color_v3 = cast_ray(&camera.eye, &rotated_direction, objects, light, 0, camera, fov, aspect_ratio);
                let pixel_color = vector3_to_color(pixel_color_v3);

                framebuffer.set_current_color(pixel_color);
                framebuffer.set_pixel(x, y);
            }
        }
    } else {
        // Lower resolution rendering with proper upscaling
        let step_x = (width as f32 / render_width as f32).ceil() as u32;
        let step_y = (height as f32 / render_height as f32).ceil() as u32;

        for y in 0..render_height {
            for x in 0..render_width {
                // Calculate the center of the block we're rendering
                let center_x = (x * step_x) + step_x / 2;
                let center_y = (y * step_y) + step_y / 2;
                
                let screen_x = (2.0 * center_x as f32) / width as f32 - 1.0;
                let screen_y = -(2.0 * center_y as f32) / height as f32 + 1.0;
                let screen_x = screen_x * aspect_ratio * perspective_scale;
                let screen_y = screen_y * perspective_scale;

                let ray_direction = Vector3::new(screen_x, screen_y, -1.0).normalized();
                let rotated_direction = camera.basis_change(&ray_direction);

                let pixel_color_v3 = cast_ray(&camera.eye, &rotated_direction, objects, light, 0, camera, fov, aspect_ratio);
                let pixel_color = vector3_to_color(pixel_color_v3);

                framebuffer.set_current_color(pixel_color);
                
                // Fill the entire block, ensuring we cover all pixels
                let start_x = x * step_x;
                let start_y = y * step_y;
                let end_x = ((x + 1) * step_x).min(width);
                let end_y = ((y + 1) * step_y).min(height);
                
                for pixel_y in start_y..end_y {
                    for pixel_x in start_x..end_x {
                        framebuffer.set_pixel(pixel_x, pixel_y);
                    }
                }
            }
        }
        
        // Fill any remaining pixels if there are gaps due to rounding
        let last_rendered_x = render_width * step_x;
        let last_rendered_y = render_height * step_y;
        
        // Fill remaining right edge
        if last_rendered_x < width {
            // Use the last column's color
            if render_width > 0 {
                let last_col_x = (render_width - 1) * step_x;
                let last_col_y = 0;
                let screen_x = (2.0 * last_col_x as f32) / width as f32 - 1.0;
                let screen_y = -(2.0 * last_col_y as f32) / height as f32 + 1.0;
                let screen_x = screen_x * aspect_ratio * perspective_scale;
                let screen_y = screen_y * perspective_scale;

                let ray_direction = Vector3::new(screen_x, screen_y, -1.0).normalized();
                let rotated_direction = camera.basis_change(&ray_direction);
                let pixel_color_v3 = cast_ray(&camera.eye, &rotated_direction, objects, light, 0, camera, fov, aspect_ratio);
                let pixel_color = vector3_to_color(pixel_color_v3);
                framebuffer.set_current_color(pixel_color);
                
                for y in 0..height {
                    for x in last_rendered_x..width {
                        framebuffer.set_pixel(x, y);
                    }
                }
            }
        }
        
        // Fill remaining bottom edge
        if last_rendered_y < height {
            // Use the last row's color
            if render_height > 0 {
                let last_row_x = 0;
                let last_row_y = (render_height - 1) * step_y;
                let screen_x = (2.0 * last_row_x as f32) / width as f32 - 1.0;
                let screen_y = -(2.0 * last_row_y as f32) / height as f32 + 1.0;
                let screen_x = screen_x * aspect_ratio * perspective_scale;
                let screen_y = screen_y * perspective_scale;

                let ray_direction = Vector3::new(screen_x, screen_y, -1.0).normalized();
                let rotated_direction = camera.basis_change(&ray_direction);
                let pixel_color_v3 = cast_ray(&camera.eye, &rotated_direction, objects, light, 0, camera, fov, aspect_ratio);
                let pixel_color = vector3_to_color(pixel_color_v3);
                framebuffer.set_current_color(pixel_color);
                
                for y in last_rendered_y..height {
                    for x in 0..last_rendered_x {
                        framebuffer.set_pixel(x, y);
                    }
                }
            }
        }
    }
}

// Create complete diorama with trees
fn create_diorama(
    piedra_texture: Image, 
    diamante_texture: Option<Image>, 
    tierra_texture: Option<Image>,
    tronco_texture: Option<Image>,
    hojas_texture: Option<Image>
) -> Vec<Cube> {
    let mut cubes = Vec::new();
    let cube_size = 1.0;
    let floor_size = 10; 
    let wall_height = 5;  
    let start_offset = -((floor_size - 1) as f32 * cube_size) / 2.0;
    
    // Materials with special properties
    let piedra_material = Material::new(
        Vector3::new(0.8, 0.8, 0.8),
        32.0,
        [0.9, 0.1, 0.0, 0.0],  // diffuse, specular, reflection, transparency
        1.0,
    );
    
    // Diamond material - highly reflective and shiny
    let diamante_material = Material::new(
        Vector3::new(0.9, 0.9, 1.0),
        128.0,
        [0.2, 0.3, 0.5, 0.0],  // Less diffuse, more reflection (50%)
        2.42,  // Diamond refractive index
    );
    
    let tierra_material = Material::new(
        Vector3::new(0.6, 0.4, 0.2),
        16.0,
        [0.9, 0.1, 0.0, 0.0],
        1.0,
    );

    let tronco_material = Material::new(
        Vector3::new(0.5, 0.3, 0.2),
        16.0,
        [0.9, 0.1, 0.0, 0.0],
        1.0,
    );

    // Leaves material - semi-transparent to let light through
    let hojas_material = Material::new(
        Vector3::new(0.2, 0.7, 0.2),
        8.0,
        [0.6, 0.1, 0.0, 0.3],  // 30% transparent to simulate leaves
        1.0,
    );
    
    // Diamond spots on floor
    let diamond_spots = vec![
        (2, 3), (7, 2), (4, 6), (8, 7)
    ];
    
    // 1. BOTTOM FLOOR (complete)
    for x in 0..floor_size {
        for z in 0..floor_size {
            let pos_x = start_offset + x as f32 * cube_size;
            let pos_z = start_offset + z as f32 * cube_size;
            let pos_y = -cube_size / 2.0;
            
            let is_diamond = diamond_spots.contains(&(x, z));
            
            let cube = if is_diamond && diamante_texture.is_some() {
                Cube::with_texture(
                    Vector3::new(pos_x, pos_y, pos_z),
                    cube_size,
                    diamante_material,
                    diamante_texture.as_ref().unwrap().clone(),
                )
            } else {
                Cube::with_texture(
                    Vector3::new(pos_x, pos_y, pos_z),
                    cube_size,
                    piedra_material,
                    piedra_texture.clone(),
                )
            };
            
            cubes.push(cube);
        }
    }
    
    // 2. WALLS (3 walls - no front wall)
    // Left wall
    for y in 0..wall_height {
        for z in 0..floor_size {
            let pos_x = start_offset;
            let pos_z = start_offset + z as f32 * cube_size;
            let pos_y = cube_size / 2.0 + y as f32 * cube_size;
            
            cubes.push(Cube::with_texture(
                Vector3::new(pos_x, pos_y, pos_z),
                cube_size,
                piedra_material,
                piedra_texture.clone(),
            ));
        }
    }
    
    // Right wall
    for y in 0..wall_height {
        for z in 0..floor_size {
            let pos_x = start_offset + (floor_size - 1) as f32 * cube_size;
            let pos_z = start_offset + z as f32 * cube_size;
            let pos_y = cube_size / 2.0 + y as f32 * cube_size;
            
            cubes.push(Cube::with_texture(
                Vector3::new(pos_x, pos_y, pos_z),
                cube_size,
                piedra_material,
                piedra_texture.clone(),
            ));
        }
    }
    
    // Back wall
    for y in 0..wall_height {
        for x in 1..(floor_size-1) {
            let pos_x = start_offset + x as f32 * cube_size;
            let pos_z = start_offset + (floor_size - 1) as f32 * cube_size;
            let pos_y = cube_size / 2.0 + y as f32 * cube_size;
            
            cubes.push(Cube::with_texture(
                Vector3::new(pos_x, pos_y, pos_z),
                cube_size,
                piedra_material,
                piedra_texture.clone(),
            ));
        }
    }
    
    // 3. TOP FLOOR - COMPLETE with ALL border cubes
    if let Some(tierra_tex) = tierra_texture {
        let top_y = cube_size / 2.0 + wall_height as f32 * cube_size;
        
        // 4x3 hole in center
        let hole_center_x = floor_size / 2;
        let hole_center_z = floor_size / 2;
        let hole_start_x = hole_center_x - 2; // 4 wide
        let hole_start_z = hole_center_z - 1; // 3 deep
        let hole_end_x = hole_start_x + 4;
        let hole_end_z = hole_start_z + 3;
        
        // Add EVERY top cube except hole
        for x in 0..floor_size {
            for z in 0..floor_size {
                let in_hole = x >= hole_start_x && x < hole_end_x && 
                             z >= hole_start_z && z < hole_end_z;
                
                if !in_hole {
                    let pos_x = start_offset + x as f32 * cube_size;
                    let pos_z = start_offset + z as f32 * cube_size;
                    
                    cubes.push(Cube::with_texture(
                        Vector3::new(pos_x, top_y, pos_z),
                        cube_size,
                        tierra_material,
                        tierra_tex.clone(),
                    ));
                }
            }
        }
        
        println!("TOP FLOOR: {} tierra cubes with complete borders", 
                 (floor_size * floor_size) - (4 * 3));
    }

    // 4. ADD MINECRAFT-STYLE TREES on top floor
    if let (Some(tronco_tex), Some(hojas_tex)) = (tronco_texture, hojas_texture) {
        let top_y = cube_size / 2.0 + wall_height as f32 * cube_size;
        
        // Tree positions - 3 trees around the hole
        let tree_positions = vec![
            (1, 1),  // Front-left of the diorama
            (8, 2),  // Front-right 
            (2, 8),  // Back-left
        ];
        
        for (tree_x, tree_z) in tree_positions {
            let tree_world_x = start_offset + tree_x as f32 * cube_size;
            let tree_world_z = start_offset + tree_z as f32 * cube_size;
            
            // TRUNK - 3 cubes tall (raised higher so it's visible)
            for trunk_height in 0..3 {
                let trunk_y = top_y + cube_size + trunk_height as f32 * cube_size;
                
                cubes.push(Cube::with_texture(
                    Vector3::new(tree_world_x, trunk_y, tree_world_z),
                    cube_size,
                    tronco_material,
                    tronco_tex.clone(),
                ));
            }
            
            // LEAVES - Start at top of trunk, raised higher
            let leaves_center_y = top_y + cube_size + 2.0 * cube_size; // Top of 3-block trunk
            
            // 3x3 leaves pattern for 2 layers only (middle and top) - no bottom layer
            for dy in 1..3 { // Start from layer 1, not 0
                for dx in -1i32..=1i32 {
                    for dz in -1i32..=1i32 {
                        let leaf_x = tree_world_x + dx as f32 * cube_size;
                        let leaf_y = leaves_center_y + (dy as f32 - 1.0) * cube_size;
                        let leaf_z = tree_world_z + dz as f32 * cube_size;
                        
                        // Create a more natural tree shape - fewer leaves on edges
                        let is_edge = dx.abs() == 1 && dz.abs() == 1;
                        let is_top_layer = dy == 2;
                        let is_center = dx == 0 && dz == 0;
                        
                        // Skip top corners for natural look
                        if is_edge && is_top_layer && !is_center {
                            continue; 
                        }
                        
                        cubes.push(Cube::with_texture(
                            Vector3::new(leaf_x, leaf_y, leaf_z),
                            cube_size,
                            hojas_material,
                            hojas_tex.clone(),
                        ));
                    }
                }
            }
            
            // Add a single crown leaf on top of the tree
            let crown_y = leaves_center_y + 1.0 * cube_size;
            cubes.push(Cube::with_texture(
                Vector3::new(tree_world_x, crown_y, tree_world_z),
                cube_size,
                hojas_material,
                hojas_tex.clone(),
            ));
        }
        
        println!("TREES: Added 3 Minecraft-style trees with elevated canopy");
        println!("Each tree: 3 trunk cubes + ~15 leaf cubes + 1 crown");
    } else {
        println!("TREES: Tronco or Hojas texture not found - skipping trees");
    }
    
    println!("TOTAL CUBES: {}", cubes.len());
    cubes
}

fn main() {
    let window_width = 800;
    let window_height = 600;
 
    let (mut window, thread) = raylib::init()
        .size(window_width, window_height)
        .title("Optimized Cave Diorama")
        .log_level(TraceLogLevel::LOG_WARNING)
        .build();

    let mut framebuffer = Framebuffer::new(window_width as u32, window_height as u32);

    // Load textures
    let piedra_paths = ["src/assets/Piedra.png", "./src/assets/Piedra.png", "./assets/Piedra.png"];
    let diamante_paths = ["src/assets/Diamante.png", "./src/assets/Diamante.png", "./assets/Diamante.png"];
    let tierra_paths = ["src/assets/Tierra.png", "./src/assets/Tierra.png", "./assets/Tierra.png"];
    let tronco_paths = ["src/assets/Tronco.png", "./src/assets/Tronco.png", "./assets/Tronco.png"];
    let hojas_paths = ["src/assets/Hojas.png", "./src/assets/Hojas.png", "./assets/Hojas.png"];

    let mut piedra_texture = None;
    for path in &piedra_paths {
        if let Ok(image) = Image::load_image(path) {
            println!("Loaded Piedra from: {}", path);
            piedra_texture = Some(image);
            break;
        }
    }

    let mut diamante_texture = None;
    for path in &diamante_paths {
        if let Ok(image) = Image::load_image(path) {
            println!("Loaded Diamante from: {}", path);
            diamante_texture = Some(image);
            break;
        }
    }

    let mut tierra_texture = None;
    for path in &tierra_paths {
        if let Ok(image) = Image::load_image(path) {
            println!("Loaded Tierra from: {}", path);
            tierra_texture = Some(image);
            break;
        }
    }

    let mut tronco_texture = None;
    for path in &tronco_paths {
        if let Ok(image) = Image::load_image(path) {
            println!("Loaded Tronco from: {}", path);
            tronco_texture = Some(image);
            break;
        }
    }

    let mut hojas_texture = None;
    for path in &hojas_paths {
        if let Ok(image) = Image::load_image(path) {
            println!("Loaded Hojas from: {}", path);
            hojas_texture = Some(image);
            break;
        }
    }

    let mut objects = if let Some(piedra) = piedra_texture {
        create_diorama(piedra, diamante_texture, tierra_texture, tronco_texture, hojas_texture)
    } else {
        println!("ERROR: Could not load Piedra texture!");
        vec![]
    };

    // Camera positioned in front of the diorama for better initial view
    let mut camera = Camera::new(
        Vector3::new(0.0, 4.0, -12.0),  // Front view, slightly elevated
        Vector3::new(0.0, 3.0, 0.0),    // Looking at center of scene
        Vector3::new(0.0, 1.0, 0.0),
    );

    // Store previous camera position for movement detection
    let mut prev_camera_pos = camera.eye;
    let mut prev_camera_angles = (camera.yaw, camera.pitch);

    // Light positioned ABOVE the hole to shine DOWN into cave
    let light = Light::new(
        Vector3::new(0.0, 10.0, 0.0),
        Color::new(255, 255, 200, 255), 
        3.0,
    );

    let movement_speed = 0.3;
    let rotation_speed = 0.03;

    println!("\n=== OPTIMIZED CAVE DIORAMA ===");
    println!("WASD: Move | Q/E: Up/Down | Arrows: Look | ESC: Exit");
    println!("OPTIMIZATIONS:");
    println!("- Adaptive rendering (lower res when moving)");
    println!("- Frustum culling (skip off-screen objects)");
    println!("- Early ray termination");
    println!("- Distance-based LOD");
    println!("- Optimized lighting calculations");

    let mut frame_count = 0;
    let mut last_fps_time = std::time::Instant::now();
    let mut frames_since_movement = 0;

    while !window.window_should_close() {
        let mut camera_moved = false;

        // Camera controls
        if window.is_key_down(KeyboardKey::KEY_W) {
            camera.move_forward(movement_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_S) {
            camera.move_forward(-movement_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_A) {
            camera.move_right(-movement_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_D) {
            camera.move_right(movement_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_Q) {
            camera.move_up(movement_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_E) {
            camera.move_up(-movement_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_LEFT) {
            camera.rotate(-rotation_speed, 0.0);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_RIGHT) {
            camera.rotate(rotation_speed, 0.0);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_UP) {
            camera.rotate(0.0, rotation_speed);
            camera_moved = true;
        }
        if window.is_key_down(KeyboardKey::KEY_DOWN) {
            camera.rotate(0.0, -rotation_speed);
            camera_moved = true;
        }

        // Detect movement for adaptive rendering
        let pos_changed = (camera.eye - prev_camera_pos).length() > 0.01;
        let angle_changed = ((camera.yaw - prev_camera_angles.0).abs() > 0.001) || 
                           ((camera.pitch - prev_camera_angles.1).abs() > 0.001);
        
        if pos_changed || angle_changed || camera_moved {
            frames_since_movement = 0;
        } else {
            frames_since_movement += 1;
        }

        // Adaptive render scale with more gradual transitions
        let render_scale = if ADAPTIVE_RENDER {
            if frames_since_movement < 3 {
                MIN_RENDER_SCALE // Very low quality while actively moving
            } else if frames_since_movement < 8 {
                MID_RENDER_SCALE // Medium quality shortly after stopping
            } else {
                MAX_RENDER_SCALE // Highest quality when still
            }
        } else {
            MAX_RENDER_SCALE
        };

        // Render with adaptive quality
        framebuffer.clear();
        render_adaptive(&mut framebuffer, &mut objects, &camera, &light, render_scale);
        framebuffer.swap_buffers(&mut window, &thread);

        // Update previous camera state
        prev_camera_pos = camera.eye;
        prev_camera_angles = (camera.yaw, camera.pitch);

        // FPS monitoring
        frame_count += 1;
        if last_fps_time.elapsed().as_secs() >= 2 {
            println!("FPS: {} | Scale: {:.2} | Cubes: {} | Pos: ({:.1}, {:.1}, {:.1})", 
                    frame_count / 2, render_scale, objects.len(), 
                    camera.eye.x, camera.eye.y, camera.eye.z);
            frame_count = 0;
            last_fps_time = std::time::Instant::now();
        }
    }
}