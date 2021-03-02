#version 130

uniform vec3 ambience;
uniform vec3 specular;
uniform vec3 diffuse;

in vec3 v_color;
in vec3 v_light_dir;
in float v_diffuse;
in vec3 v_pos;

out vec4 out_color;

void main() {
    float f_diffuse = v_diffuse;
    float f_specular = max(pow(dot(normalize(v_pos), v_light_dir), 3), 0);
    out_color = vec4((ambience + f_specular * specular + f_diffuse * diffuse) * v_color, 1);
}