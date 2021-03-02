#version 130

uniform mat4 mvp;
uniform vec4 tint;

in vec3 pos;
in uint color;

out vec4 v_color;

void main() {
    //Extract R, G, B and normal index from a single 32-bit input
    float r = float((color >> 24u) & 0xffu) * (1./256.);
    float g = float((color >> 16u) & 0xffu) * (1./256.);
    float b = float((color >> 8u) & 0xffu) * (1./256.);
    float a = float(color & 0xffu) * (1./256.);
    //vec3 raw_normal = NORMALS[color & 0xffu];

    v_color = vec4(r, g, b, a) * tint;
    gl_Position = mvp * vec4(pos, 1);
}
