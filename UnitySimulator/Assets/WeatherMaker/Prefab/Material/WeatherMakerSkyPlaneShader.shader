Shader "WeatherMaker/WeatherMakerSkyPlaneShader"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "black" {}
		_NightTex("Night Texture", 2D) = "black" {}
	}
	SubShader
	{
		Tags { "Queue" = "Transparent" "RenderType" = "Background" "IgnoreProjector" = "True" "PerformanceChecks" = "False" "LightMode" = "Always" }
		Cull Off ZWrite Off ZTest Always Fog { Mode Off } Blend Off

		Pass
		{
			CGPROGRAM

			#pragma target 3.0
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile __ ENABLE_NIGHT_TWINKLE

			#include "WeatherMakerSkyShader.cginc"

			fixed _WeatherMakerSkyYOffset;

			v2fSky vert (appdata v)
			{
				v2fSky o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv; // TRANSFORM_TEX not supported
				o.ray = lerp(float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 0.0), max(0.0, o.uv.y - _WeatherMakerSkyYOffset));
				procedural_sky_info i = CalculateScatteringCoefficients(_WeatherMakerSunDirectionDown2D, _WeatherMakerSunColor.rgb * pow(_WeatherMakerSunColor.a, 0.5), 1.0, normalize(o.ray));
				o.inScatter = i.inScatter;
				o.outScatter = i.outScatter;
                return o;
			}
			
			fixed4 frag (v2fSky i) : SV_Target
			{
				i.ray = normalize(i.ray);
				fixed3 sunColor = _WeatherMakerSunColor.rgb * _WeatherMakerSunColor.a;
				procedural_sky_info p = CalculateScatteringColor(_WeatherMakerSunDirectionDown2D, sunColor, 0.0, i.ray, i.inScatter, i.outScatter);
				fixed3 nightColor = GetNightColor(i.ray, i.uv);
				fixed3 result = ((p.inScatter + p.outScatter) * _WeatherMakerSkyGradientColor) + nightColor;
				ApplyDither(result.rgb, i.uv, _WeatherMakerSkyDitherLevel);
				return float4(result.rgb, 1.0);
			}

			ENDCG
		}
	}
}
