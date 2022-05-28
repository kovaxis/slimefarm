#version 130

uniform vec3 base;
uniform vec3 lowest;
uniform vec3 highest;
uniform vec3 sunrise;
uniform vec3 sun_dir;
uniform float cycle;

in vec3 v_pos;

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

// sun fireball
vec3 fireball(vec3 dir) {
    float sun = 0.04 / length(dir - sun_dir);
    sun *= smoothstep(-0.15, -0.02, sun_dir.z);
    vec3 sunball = vec3(1, 1, 1);
    return min(vec3(1), vec3(1, 1, 0.9) * sun);
}

float star(vec3 p) {
    float d = length(p);
    float m = 0.0001 / d;
    m *= m;
    //m *= smoothstep(0.8, 0.2, d);
    return m;
}

const float TAU = 6.28318530718;
const float PHI = 2.39996322973;
const float STARS = 512.;
const float LAYERS = 4.;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

vec3 starlayer(vec3 p, float l) {
    vec3 m = vec3(0.);
    for(float i = 1.; i < STARS; i += 1.) {
        float y = 1. - i * (2. / STARS);
        float r = sqrt(1. - y*y);
        float theta = PHI * i;
        float x = sin(theta) * r;
        float z = -cos(theta) * r;
        vec3 starpos = vec3(x, y, z);

        float rnd = rand(vec2(i, 19.32 + l));
        vec3 off = vec3(rnd, fract(rnd * 12.492), fract(rnd * 288.928)) - .5;
        starpos += off * 0.3;
        starpos = normalize(starpos);

        float size = 0.02 + fract(rnd * 37.58) * 0.4;
        size *= (l + 1.) * 7.;
        size *= size;

        vec3 color = sin(fract(rnd * 49.283) * 1248.81 * vec3(0.34, 0.19, 0.79));
        color = vec3(.9, .9, .9) + color * vec3(0.1, 0., 0.1);

        m += star(p - starpos) * size * color;
    }
    return m;
}

// stars
vec3 starfield(vec3 p) {
    vec3 m = vec3(0.);

    for(float l = 0.; l < 1.; l+= 1. / LAYERS) {
        m += starlayer(p, l);
    }
    
    m *= smoothstep(-0.1, -0.25, sun_dir.z);
    return m;
}

// get sky color
vec3 get_sky(vec3 p) {
    vec3 color = vec3(0);

    color += background(p);
    color += fireball(p);
    //color += starfield(p);

    return color;
}

void main() {
    vec3 p = normalize(v_pos);
    out_color = vec4(get_sky(p), 1);

    // TODO: Change the sky appearance depending on the fog distance
    // Near fog should have a more monotone color
    // Far fog has more "sky" in it, with a marked horizon and directional features
}
