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

// Resources:
// http://library.nd.edu/documents/arch-lib/Unity/SB/Assets/SampleScenes/Shaders/Skybox-Procedural.shader
//

Shader "WeatherMaker/WeatherMakerSkySphereShader"
{
	Properties
	{
		_MainTex("Day Texture", 2D) = "blue" {}
		_DawnDuskTex("Dawn/Dusk Texture", 2D) = "orange" {}
		_NightTex("Night Texture", 2D) = "black" {}
		_DayMultiplier("Day Multiplier", Range(0, 3)) = 1
		_DawnDuskMultiplier("Dawn/Dusk Multiplier", Range(0, 1)) = 0
		_NightMultiplier("Night Multiplier", Range(0, 1)) = 0
		_NightSkyMultiplier("Night Sky Multiplier", Range(0, 1)) = 0
		_NightVisibilityThreshold("Night Visibility Threshold", Range(0, 1)) = 0
		_NightIntensity("Night Intensity", Range(0, 32)) = 2
		_NightTwinkleSpeed("Night Twinkle Speed", Range(0, 100)) = 16
		_NightTwinkleVariance("Night Twinkle Variance", Range(0, 10)) = 1
		_NightTwinkleMinimum("Night Twinkle Minimum Color", Range(0, 1)) = 0.02
		_NightTwinkleRandomness("Night Twinkle Randomness", Range(0, 5)) = 0.15
	}
	SubShader
	{
		Tags { "Queue" = "Geometry+497" "RenderType" = "Opaque" "IgnoreProjector" = "True" "PerformanceChecks" = "False" "PreviewType" = "Skybox" }

		CGINCLUDE

		#include "WeatherMakerSkyShader.cginc"

		#pragma target 3.0
		#pragma fragmentoption ARB_precision_hint_fastest
		#pragma glsl_no_auto_normalization
		#pragma multi_compile __ UNITY_HDR_ON
		#pragma multi_compile ENABLE_TEXTURED_SKY ENABLE_PROCEDURAL_TEXTURED_SKY ENABLE_PROCEDURAL_SKY
		#pragma multi_compile __ ENABLE_NIGHT_TWINKLE

		v2fSky vert(appdata v)
		{
			v2fSky o;
			o.vertex = UnityObjectToClipPosFarPlane(v.vertex);
			o.uv.xy = v.uv; // TRANSFORM_TEX not supported
			o.ray = WorldSpaceVertexPos(v.vertex) - _WorldSpaceCameraPos;
			procedural_sky_info i = CalculateScatteringCoefficients(_WeatherMakerSunDirectionUp, _WeatherMakerSunColor.rgb * _WeatherMakerSunVar1.y, 1.0, normalize(o.ray));
			o.inScatter = i.inScatter;
			o.outScatter = i.outScatter;
			return o;
		}

		fixed4 fragBase(v2fSky i)
		{
			fixed4 result;
			i.ray = normalize(i.ray);
			procedural_sky_info p = CalculateScatteringColor(_WeatherMakerSunDirectionUp, _WeatherMakerSunColor.rgb, _WeatherMakerSunVar1.x, i.ray, i.inScatter, i.outScatter);
			fixed sunMoon;
			fixed4 skyColor = p.skyColor;
			skyColor.rgb *= _WeatherMakerSkyGradientColor;
			fixed3 nightColor = GetNightColor(i.ray, i.uv);

#if defined(ENABLE_PROCEDURAL_TEXTURED_SKY)

			fixed4 dayColor = tex2D(_MainTex, i.uv) * _DayMultiplier;
			fixed4 dawnDuskColor = tex2D(_DawnDuskTex, i.uv);
			fixed4 dawnDuskColor2 = dawnDuskColor * _DawnDuskMultiplier;
			dayColor += dawnDuskColor2;

			// hide night texture wherever dawn/dusk is opaque, reduce if clouds
			nightColor *= (1.0 - dawnDuskColor.a);

			// blend texture on top of sky
			result = ((dayColor * dayColor.a) + (skyColor * (1.0 - dayColor.a)));

			// blend previous result on top of night
			result = ((result * result.a) + (fixed4(nightColor, 1.0) * (1.0 - result.a)));

#elif defined(ENABLE_PROCEDURAL_SKY)

			result = skyColor + fixed4(nightColor, 0.0);

#else

			fixed4 dayColor = tex2D(_MainTex, i.uv) * _DayMultiplier;
			fixed4 dawnDuskColor = (tex2D(_DawnDuskTex, i.uv) * _DawnDuskMultiplier);
			result = (dayColor + dawnDuskColor + fixed4(nightColor, 0.0));

#endif

			ApplyDither(result.rgb, i.uv, _WeatherMakerSkyDitherLevel);
			return result;
		}

		fixed4 frag(v2fSky i) : SV_Target
		{
			return fragBase(i);
		}

		ENDCG

		Pass
		{
			Tags { "LightMode" = "Always" }
			Cull Front Lighting Off ZWrite Off ZTest LEqual
			Blend [_SrcBlendMode] [_DstBlendMode]

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag

			ENDCG
		}
	}

	/*
	Subshader
	{
		Tags { "Queue" = "Geometry+497" "RenderType" = "Opaque"	"IgnoreProjector" = "True" }

		// http://answers.unity3d.com/questions/973067/using-shadow-texture-to-recieve-shadows-on-grass.html
		Pass
        { 
			Name "SkySphereCloudShadowCasterPass"
			Tags { "LightMode" = "ShadowCaster" }
			ZWrite On ZTest Less Cull Back Fog { Mode Off }
			Offset 1, 1

			CGPROGRAM

			#pragma target 3.0
			#pragma vertex shadowVert2
			#pragma fragment shadowFrag2
			#pragma fragmentoption ARB_precision_hint_fastest
			#pragma multi_compile_shadowcaster
			#pragma glsl_no_auto_normalization

			struct v2fShadow2
			{
				V2F_SHADOW_CASTER;
				float3 ray : NORMAL;
			};
 
			v2fShadow2 shadowVert2(appdata_full v)
			{
				v2fShadow2 o;
				o.ray = WorldSpaceVertexPos(v.vertex) - _WorldSpaceCameraPos;
				TRANSFER_SHADOW_CASTER(o); 
				//TRANSFER_SHADOW_CASTER_NORMALOFFSET(o);
				return o;
			}
 
			float4 shadowFrag2(v2fShadow2 i) : COLOR
			{

#if defined(ENABLE_CLOUDS) || defined(ENABLE_CLOUDS_MASK)

				float3 worldPos;
				fixed f = ComputeCloudFBM(normalize(i.ray), worldPos);
				clip(f - _FogShadowThreshold);
				SHADOW_CASTER_FRAGMENT(i);

#else

				clip(-1.0);
				SHADOW_CASTER_FRAGMENT(i);

#endif

			}

			ENDCG
        }
	}
	*/

	FallBack Off
}
