#version 330 core
layout (location = 0) in vec3 aPos;

uniform float rotation;

vec3 rotate(vec3 pos, float angle) {
    float x = pos.x * cos(angle) - pos.y * sin(angle);
    float y = pos.x * sin(angle) + pos.y * cos(angle);
    return vec3(x, y, pos.z);
}

void main() {
    gl_Position = vec4(rotate(aPos, rotation), 1.0);
}