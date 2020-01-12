Shader "WeatherMaker/WeatherMakerFullScreenCloudsShader"
{
	Properties
	{
		_FogDensity("Fog Density", Range(0.0, 1.0)) = 0.0
		_FogColor("Fog Color", Color) = (1,1,1,1)
		_FogEmissionColor("Fog Emission Color", Color) = (0,0,0,0)
		_FogNoise("Fog Noise", 2D) = "white" {}
		_FogNoise2("Fog Noise2", 2D) = "clear" {}
		_FogNoiseScale("Fog Noise Scale", Range(0.0, 1.0)) = 0.02
		_FogNoiseMultiplier("Fog Noise Multiplier", Range(0.01, 4.0)) = 1
		_FogNoiseVelocity("Fog Noise Velocity", Vector) = (0.1, 0.2, 0, 0)
		_FogNoiseMask("Fog Noise Mask", 2D) = "white" {}
		_FogNoiseMaskScale("Fog Noise Mask Scale", Range(0.000001, 1.0)) = 0.02
		_FogNoiseMaskOffset("Fog Noise Mask Offset", Vector) = (0.0, 0.0, 0.0)
		_FogNoiseMaskVelocity("Fog Noise Mask Velocity", Vector) = (0.1, 0.2, 0, 0)
		_FogNoiseMaskRotationSin("Fog Noise Mask Rotation Sin", Float) = 0.0
		_FogNoiseMaskRotationCos("Fog Noise Mask Rotation Cos", Float) = 0.0
		_FogCover("Fog Cover", Range(0.0, 1.0)) = 0.25
		_FogLightAbsorption("Fog Light Absorption", Range(0.0, 1.0)) = 0.013
		_FogSharpness("Fog Sharpness", Range(0.0, 1.0)) = 0.015
		_FogWhispiness("Fog Whispiness", Range(0.0, 3.0)) = 1.0
		_FogWhispinessChangeFactor("Fog Whispiness Change Factor", Range(0.0, 1.0)) = 0.03
		_PointSpotLightMultiplier("Point/Spot Light Multiplier", Range(0, 10)) = 1
		_DirectionalLightMultiplier("Directional Light Multiplier", Range(0, 10)) = 1
		_AmbientLightMultiplier("Ambient light multiplier", Range(0, 4)) = 1
	}
	SubShader
	{
		Tags{ "Queue" = "Geometry+503" "IgnoreProjector" = "True" "RenderType" = "Transparent" "LightMode" = "Always" }
		Cull Back Lighting Off ZWrite Off ZTest LEqual Fog { Mode Off }
		Blend [_SrcBlendMode][_DstBlendMode]

		Pass
		{
			CGPROGRAM

			#pragma vertex full_screen_vertex_shader
			#pragma fragment frag
			#pragma multi_compile __ ENABLE_CLOUDS ENABLE_CLOUDS_MASK
			#pragma multi_compile __ UNITY_MULTI_PASS_STEREO
			#pragma multi_compile __ UNITY_HDR_ON
			
			#include "WeatherMakerSkyShader.cginc"

			fixed4 frag (full_screen_fragment i) : SV_Target
			{

#if defined(ENABLE_CLOUDS) || defined(ENABLE_CLOUDS_MASK)

				float3 cloudRay = normalize(i.forwardLine);
				fixed4 inScatter = _WeatherMakerSunColor;// fixed4(1.0, 1.0, 1.0, 1.0);
				float3 worldPos;
				cloudRay.y += _WeatherMakerCloudRayOffset;
				fixed sunAngleAmount = (1.0 - pow(_WeatherMakerSunColor.a * 0.7, 8.0)) * pow(1.0 - abs(max(0.0, min(1.0, _WeatherMakerSunDirectionUp.y + 0.04))), 8.0);
				fixed4 sunLightColor =  lerp(_WeatherMakerSunColor, inScatter, sunAngleAmount);
				fixed4 cloudColor = ComputeCloudColor(cloudRay, sunLightColor, worldPos);
				cloudColor.rgb *= min(1.0, cloudColor.a * 1.5);
				ApplyDither(cloudColor.rgb, i.uv, _WeatherMakerSkyDitherLevel);
				return cloudColor;

#else

				return fixed4(0.0, 0.0, 0.0, 0.0);

#endif

			}

			ENDCG
		}
	}
}
