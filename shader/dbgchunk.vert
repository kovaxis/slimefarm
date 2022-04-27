#version 130

uniform vec3 offset;
uniform mat4 mvp;

in vec3 pos;
in vec3 normal;
in vec4 color;

out vec4 v_color;

void main() {
    v_color = color;
    gl_Position = mvp * vec4(pos + offset, 1);
}
