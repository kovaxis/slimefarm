#version 130

uniform mat4 mvp;
uniform vec3 offset;

uniform mat4 view;
in vec3 pos;
in uint color;

out vec3 v_pos;

void main() {
    vec4 pos_w = vec4(pos + offset, 1);
    vec4 scr_pos_w = mvp * pos_w;
    float iw = 1. / scr_pos_w.w;
    vec3 scr_pos = scr_pos_w.xyz * iw;

    v_pos = vec3(view * vec4(scr_pos.x, 1, scr_pos.y, 0));
    gl_Position = vec4(scr_pos.xy, 1, 1);
}
