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

// https://alastaira.wordpress.com/2014/12/30/adding-shadows-to-a-unity-vertexfragment-shader-in-7-easy-steps/

Shader "WeatherMaker/WeatherMakerMoonShader"
{
	Properties
	{
		_MainTex("Moon Texture", 2D) = "white" {}
	}
	SubShader
	{
		Tags { "Queue" = "Geometry+499" "RenderType" = "Background" "IgnoreProjector" = "True" "PerformanceChecks" = "False" "LightMode" = "Always" }
		Cull Back Lighting Off ZWrite Off ZTest LEqual Blend SrcAlpha OneMinusSrcAlpha

		CGINCLUDE

		#include "WeatherMakerSkyShader.cginc"

		#pragma target 3.0
		#pragma fragmentoption ARB_precision_hint_fastest
		#pragma glsl_no_auto_normalization
		#pragma multi_compile __ UNITY_HDR_ON
		#pragma multi_compile __ ENABLE_SUN_ECLIPSE

		sampler2D _WeatherMakerSkySphereTexture;
		int _MoonIndex;

		struct vertexInput
		{
			float4 vertex: POSITION;
			float4 normal: NORMAL;
			float4 texcoord: TEXCOORD0;
		};

		struct vertexOutput
		{
			float4 pos: SV_POSITION;
			float3 normalWorld: NORMAL;
			float2 tex: TEXCOORD0;
			float4 grabPos: TEXCOORD1;
			float3 ray : TEXCOORD2;
		};

		vertexOutput vert(vertexInput v)
		{
			vertexOutput o;
			o.normalWorld = normalize(WorldSpaceVertexPosNear(v.normal));
			o.pos = UnityObjectToClipPos(v.vertex);
			o.tex = TRANSFORM_TEX(v.texcoord, _MainTex);
			o.grabPos = ComputeGrabScreenPos(o.pos);
			o.ray = WorldSpaceVertexPos(v.vertex) - _WorldSpaceCameraPos;

			return o;
		}

		fixed4 frag(vertexOutput i) : SV_TARGET
		{
			i.ray = normalize(i.ray);
			fixed nightMultiplier = max(0.0, 1.0 - _WeatherMakerSunColor.a);

#if defined(ENABLE_SUN_ECLIPSE)

			fixed lerpSun = CalcSunSpot(_WeatherMakerSunVar1.x * 1.6, _WeatherMakerSunDirectionUp, i.ray);
			fixed feather = saturate(dot(-i.ray, i.normalWorld) * 3.0);
			fixed4 moonColor = fixed4(0.0, 0.0, 0.0, lerpSun * feather);

#else

			fixed4 moonColor = tex2D(_MainTex, i.tex.xy);
			fixed lightNormal = max(0.0, dot(i.normalWorld, _WeatherMakerSunDirectionUp));
			fixed3 lightFinal = _WeatherMakerSunColor.rgb * lightNormal * _WeatherMakerMoonTintColor[_MoonIndex].rgb * _WeatherMakerMoonTintColor[_MoonIndex].a;
			fixed lightMax = max(lightFinal.r, max(lightFinal.g, lightFinal.b));
			fixed feather = pow(abs(dot(i.ray, i.normalWorld)), 0.2);

			// alpha ramps up as night approaches or is the maximum light value
			moonColor.a = max(pow(nightMultiplier, 4), lightMax);

			// reduce alpha during the day, at full night no alpha reduction
			moonColor.a *= (pow(nightMultiplier, 4) + 0.5) * 0.6667 * feather;

			// apply sun light
			moonColor.rgb *= lightFinal;

#endif

			return moonColor;
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