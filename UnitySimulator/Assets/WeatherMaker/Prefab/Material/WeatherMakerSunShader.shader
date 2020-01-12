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

Shader "WeatherMaker/WeatherMakerSunShader"
{
	Properties
	{
	}
	SubShader
	{
		Tags { "Queue" = "Geometry+498" "RenderType" = "Background" "IgnoreProjector" = "True" "PerformanceChecks" = "False" "LightMode" = "Always" }
		Cull Front Lighting Off ZWrite Off ZTest LEqual
        Blend One One

		CGINCLUDE

		#include "WeatherMakerSkyShader.cginc"

		#pragma target 3.0
		#pragma fragmentoption ARB_precision_hint_fastest
		#pragma glsl_no_auto_normalization
		#pragma multi_compile __ UNITY_HDR_ON
		#pragma multi_compile __ RENDER_HINT_FAST

		struct v2fSun
		{
			float4 vertex : SV_POSITION;
			float3 ray : NORMAL;
			float2 uv : TEXCOORD0;
		};

		v2fSun vert (appdata v)
		{
			v2fSun o;
			o.vertex = UnityObjectToClipPos(v.vertex);
			o.ray = WorldSpaceVertexPos(v.vertex) - _WorldSpaceCameraPos;
			o.uv = v.uv;
			return o;
		}

		fixed4 fragBase(v2fSun i)
		{
			i.ray = normalize(i.ray);

#if defined(RENDER_HINT_FAST)

			fixed miePhase = CalcSunSpot(_WeatherMakerSunVar1.x, _WeatherMakerSunDirectionUp, i.ray);

#else

			float eyeCos = dot(_WeatherMakerSunDirectionUp, i.ray);
			float eyeCos2 = eyeCos * eyeCos;

			// get mie phase, and power it out at the edges to avoid artifacts when overlaying on the sky sphere
			fixed miePhase = GetMiePhase(_WeatherMakerSunVar1.x, eyeCos, eyeCos2, 1.18);

#endif

			fixed4 sunColor = fixed4(_WeatherMakerSunColor.rgb * _WeatherMakerSunTintColor.rgb * miePhase, miePhase);
			ApplyDither(sunColor.rgb, i.uv, _WeatherMakerSkyDitherLevel);
			return sunColor;
		}
			
		fixed4 frag (v2fSun i) : SV_TARGET
		{
			return fragBase(i);
		}

		ENDCG

		Pass
		{
			Tags { "LightMode" = "ForwardBase" }

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag

			ENDCG
		}
	}

	FallBack Off
}
