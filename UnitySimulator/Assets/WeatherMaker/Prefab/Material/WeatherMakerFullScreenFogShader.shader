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

Shader "WeatherMaker/WeatherMakerFullScreenFogShader"
{
	Properties
	{
		_FogColor("Fog Color", Color) = (0,1,1,1)
		_FogNoise("Fog Noise", 2D) = "white" {}
		_FogNoiseScale("Fog Noise Scale", Range(0.0, 1.0)) = 0.0005
		_FogNoiseMultiplier("Fog Noise Multiplier", Range(0.01, 1.0)) = 0.15
		_FogNoiseVelocity("Fog Noise Velocity", Vector) = (0.01, 0.01, 0, 0)
		_FogDensity("Fog Density", Range(0.0, 1.0)) = 0.05
		_FogHeight("Fog Height", Float) = 0
		_MaxFogFactor("Maximum Fog Facto", Range(0.01, 1)) = 1
		_FarPlaneSunThreshold("Far Plane Sun Threshold", Range(0, 1)) = 0.75
		_FogDitherLevel("Fog Dither Level", Range(0, 1)) = 0.005
		_PointSpotLightMultiplier("Point/Spot Light Multiplier", Range(0, 10)) = 1
		_DirectionalLightMultiplier("Directional Light Multiplier", Range(0, 10)) = 1
		_AmbientLightMultiplier("Ambient Light Multiplier", Range(0, 10)) = 2
	}
	Category
	{
		Tags{ "Queue" = "Geometry+504" "IgnoreProjector" = "True" "RenderType" = "Transparent" "LightMode" = "Always" }
		Cull Back Lighting Off ZWrite Off ZTest Always Fog { Mode Off }

		SubShader
		{
			Pass
			{
				Blend[_SrcBlendMode][_DstBlendMode]

				CGPROGRAM

				#include "WeatherMakerFogShader.cginc"

				#pragma target 3.0
				#pragma vertex full_screen_vertex_shader
				#pragma fragment fog_box_full_screen_fragment_shader
				#pragma fragmentoption ARB_precision_hint_fastest
				#pragma glsl_no_auto_normalization
				#pragma multi_compile __ ENABLE_FOG_NOISE
				#pragma multi_compile __ ENABLE_FOG_HEIGHT
				#pragma multi_compile __ WEATHER_MAKER_FOG_EXPONENTIAL WEATHER_MAKER_FOG_LINEAR WEATHER_MAKER_FOG_EXPONENTIAL_SQUARED WEATHER_MAKER_FOG_CONSTANT
				#pragma multi_compile __ ENABLE_FOG_LIGHTS ENABLE_FOG_LIGHTS_WITH_SHADOWS
				#pragma multi_compile __ ENABLE_FOG_SUN_SHAFTS
				#pragma multi_compile __ UNITY_MULTI_PASS_STEREO
				#pragma multi_compile __ SHADOWS_ONE_CASCADE
				#pragma multi_compile __ WEATHER_MAKER_FOG_ENABLE_NULL_FOG_ZONES

				ENDCG
			}
		}
	}
	Fallback "VertexLit"
}
