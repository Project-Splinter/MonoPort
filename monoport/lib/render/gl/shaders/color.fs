#version 330

out vec4 FragColor;

in vec3 PassColor;

void main()
{
	FragColor = vec4(PassColor, 1.0);
}
