#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TextureCoord;

// All in camera space
out VertexData {
    vec3 Position;
    vec3 Normal;
    vec2 TextureCoord;
} VertexOut;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

uniform int UVMode = 0;


void main()
{
    
    VertexOut.Position = a_Position;
    VertexOut.TextureCoord = a_TextureCoord;
	if(UVMode > 0) {
        VertexOut.Normal = a_Normal.xyz;
        gl_Position = vec4(a_TextureCoord, 0.0, 1.0) - vec4(0.5, 0.5, 0, 0);
        gl_Position[0] *= 2.0;
        gl_Position[1] *= 2.0;
    }
    else {
        VertexOut.Normal = (ModelMat * vec4(a_Normal, 0.0)).xyz;
        gl_Position = PerspMat * ModelMat * vec4(a_Position, 1.0);
    }
}