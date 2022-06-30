#version 130

uniform sampler2D color;
uniform sampler2D light;

uniform vec3 tint;
uniform vec3 ambience;
uniform vec3 specular;
uniform vec3 diffuse;
uniform float fog;

uniform vec3 base;
uniform vec3 lowest;
uniform vec3 highest;
uniform vec3 sunrise;
uniform vec3 sun_dir;
uniform float cycle;

smooth in vec2 v_cuv;
smooth in vec2 v_luv;

smooth in float v_diffuse;
smooth in vec3 v_light_dir;
smooth in vec3 v_pos;
smooth in float v_dist;

out vec4 out_color;

// base sky color
vec3 background(vec3 dir) {
    vec3 highest_d = highest - base;
    vec3 lowest_d = lowest - base;
    float altitude = pow(max(dir.z, 0), 1);
    float decline = pow(max(-dir.z, 0), 1) * 0.4;
    float east = max(2 * dot(dir, sun_dir) - 4 * max(dir.z, -0.2), 0);
    return base + highest_d * altitude + lowest_d * decline + sunrise * east;
}

// compute fog from distance to camera.
float get_fog(float dist) {
    float x = dist / fog;
    x *= x;
    x *= x;
    x *= x;
    return clamp(1.01 - x, 0., 1.);
    //float d = 0.001;
    //d = -d*d;
    //float alpha = (exp(d * (x * x)) - exp(d)) * (1. / (1. - exp(d)));
    //float alpha = 1 - clamp(pow(2., dist / fog) - 1., 0., 1.);
    //float alpha = smoothstep(fog, fog * 0.4, dist);
    //float alpha = pow(clamp((1 / 7.6) * (fog - dist), 0, 1), 2);
    //return max(0., alpha);
}

void main() {
    float gamma = 2.2;

    // Extract raw color and light from atlas textures
    vec4 rawc = texture2D(color, v_cuv);
    vec4 rawl = texture2D(light, v_luv);

    // Extract base color
    vec3 basecolor = rawc.rgb;

    // Extract shininess
    float shininess = rawc.a * rawc.a;

    // Extract ambient occlusion
    float ao = pow(rawl.x, gamma);

    // Extract sky lighting
    float sky = pow(rawl.y, gamma);

    // Compute fog alpha
    vec3 v_dir = v_pos / v_dist;
    float alpha = get_fog(v_dist);

    // Compute light color
    float w_diffuse = v_diffuse;
    float w_specular = max(pow(dot(v_dir, v_light_dir), 3), 0) * shininess;
    vec3 lighting = (ambience + w_diffuse * diffuse + w_specular * specular) * sky;

    // Compute final color
    vec3 color = basecolor * ao * lighting + tint;
    vec3 fog_color = background(v_dir);
    color = mix(fog_color, color, alpha);



    out_color = vec4(color, 1);
}
