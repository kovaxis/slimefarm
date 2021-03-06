#version 130

uniform vec3 base;
uniform vec3 lowest;
uniform vec3 highest;
uniform vec3 sunrise;
uniform vec3 sunrise_dir;

in vec3 v_pos;

out vec4 out_color;

void main() {
    vec3 dir = normalize(v_pos);
    vec3 highest_d = highest - base;
    vec3 lowest_d = lowest - base;
    float altitude = pow(max(dir.y, 0), 1);
    float decline = pow(max(-dir.y, 0), 1) * 0.4;
    float east = max(2 * dot(dir, sunrise_dir) - 4 * max(dir.y, -0.2), 0);
    out_color = vec4(base + highest_d * altitude + lowest_d * decline + sunrise * east, 1);
}
