#version 130

uniform vec3 ambience;
uniform vec3 specular;
uniform vec3 diffuse;
uniform float fog;

in vec3 v_color;
in vec3 v_light_dir;
in float v_diffuse;
in vec3 v_pos;

out vec4 out_color;

void main() {
    float dist = length(v_pos);
    float alpha = pow(clamp((1 / 7.6) * (fog - dist), 0, 1), 2);
    if (alpha < 0.01) {
        discard;
    }
    float inv_dist = 1 / dist;
    float f_diffuse = v_diffuse;
    float f_specular = max(pow(dot(v_pos * inv_dist, v_light_dir), 3), 0);
    out_color = vec4((ambience + f_specular * specular + f_diffuse * diffuse) * v_color, alpha);
}