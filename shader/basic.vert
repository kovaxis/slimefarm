#version 130

uniform mat4 mvp;
uniform vec4 tint;

in vec3 pos;
in vec3 normal;
in vec4 color;

out vec4 v_color;

void main() {
    v_color = color * tint;
    gl_Position = mvp * vec4(pos, 1);
}
