#version 130

uniform mat4 view;

in vec3 pos;
in uint color;

out vec3 v_pos;

void main() {
    v_pos = vec3(view * vec4(pos, 0));
    gl_Position = vec4(pos, 1);
}