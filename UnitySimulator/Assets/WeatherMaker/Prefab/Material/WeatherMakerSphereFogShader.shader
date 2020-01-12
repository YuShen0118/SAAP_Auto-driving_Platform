//
// Weather Maker for Unity
// (c) 2016 Digital Ruby, LLC
// Source code may be used for personal or commercial projects.
// Source code may NOT be redistributed or sold.
// 
// *** A NOTE ABOUT PIRACY ***
// 
// If you got this asset off of leak forums or any other horrible evil pirate site, please consider buying it from the Unity asset store at https ://www.assetstore.unity3d.com/en/#!/content/60955?aid=1011lGnL. This asset is only legally available from the Unity Asset Store.
// 
// I'm a single indie dev supporting my family by spending hundreds and thousands of hours on this and other assets. It's very offensive, rude and just plain evil to steal when I (and many others) put so much hard work into the software.
// 
// Thank you.
//
// *** END NOTE ABOUT PIRACY ***
//

Shader "WeatherMaker/WeatherMakerBoxFogShader"
{
	Properties
	{
		_FogColor("Fog Color", Color) = (0,1,1,1)
		_FogNoise("Fog Noise", 2D) = "white" {}
		_FogNoiseScale("Fog Noise Scale", Range(0.0, 1.0)) = 0.0005
		_FogNoiseMultiplier("Fog Noise Multiplier", Range(0.01, 1.0)) = 0.15
		_FogNoiseVelocity("Fog Noise Velocity", Vector) = (0.01, 0.01, 0, 0)
		_FogDensity("Fog Density", Range(0.0, 1.0)) = 0.05
		_FogSpherePosition("Fog Sphere Position (x,y,z,radius squared)", Vector) = (0, 0, 0, 0)
		_MaxFogFactor("Maximum Fog Factor", Range(0.01, 1)) = 1
		_PointSpotLightMultiplier("Point/Spot Light Multiplier", Range(0, 10)) = 1
		_DirectionalLightMultiplier("Directional Light Multiplier", Range(0, 10)) = 1
		_AmbientLightMultiplier("Ambient Light Multiplier", Range(0, 10)) = 2
	}
	Category
	{
		Tags{ "Queue" = "Geometry+504" "IgnoreProjector" = "True" "RenderType" = "Transparent" "LightMode" = "Always" }
		Cull Front Lighting Off ZWrite Off ZTest Always Fog { Mode Off }
		ColorMask RGBA
		Blend One OneMinusSrcAlpha

		SubShader
		{
			Pass
			{
				CGPROGRAM

				#pragma target 3.0
				#pragma vertex fog_volume_vertex_shader
				#pragma fragment fog_sphere_fragment_shader
				#pragma fragmentoption ARB_precision_hint_fastest
				#pragma glsl_no_auto_normalization
				#pragma multi_compile __ ENABLE_FOG_NOISE
				#pragma multi_compile __ WEATHER_MAKER_FOG_EXPONENTIAL WEATHER_MAKER_FOG_LINEAR WEATHER_MAKER_FOG_EXPONENTIAL_SQUARED WEATHER_MAKER_FOG_CONSTANT
				#pragma multi_compile __ ENABLE_FOG_LIGHTS ENABLE_FOG_LIGHTS_WITH_SHADOWS
				#pragma multi_compile __ SHADOWS_ONE_CASCADE

				#include "WeatherMakerFogShader.cginc"

				ENDCG
			}
		}
	}
	Fallback "VertexLit"
}
