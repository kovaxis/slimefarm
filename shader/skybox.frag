#version 130

uniform vec3 base;
uniform vec3 lowest;
uniform vec3 highest;
uniform vec3 sunrise;
uniform vec3 sun_dir;

in vec3 v_pos;

out vec4 out_color;

void main() {
    vec3 dir = normalize(v_pos);
    vec3 highest_d = highest - base;
    vec3 lowest_d = lowest - base;
    float altitude = pow(max(dir.z, 0), 1);
    float decline = pow(max(-dir.z, 0), 1) * 0.4;
    float east = max(2 * dot(dir, sun_dir) - 4 * max(dir.z, -0.2), 0);
    float sun = max((dot(dir, sun_dir) - 0.9) * 10, 0);
    sun = sun * sun;
    vec3 sunball = vec3(1, 1, 1);
    out_color = vec4(base + highest_d * altitude + lowest_d * decline + sunrise * east + sunball * sun, 1);

    // TODO: Change the sky appearance depending on the fog distance
    // Near fog should have a more monotone color
    // Far fog has more "sky" in it, with a marked horizon and directional features
}
