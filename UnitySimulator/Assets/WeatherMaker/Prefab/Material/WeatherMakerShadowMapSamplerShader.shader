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

Shader "WeatherMaker/WeatherMakerShadowMapSamplerShader"
{
	Properties { _MainTex ("Texture", any) = "" {} }
    SubShader
	{
		ZTest Always Cull Back ZWrite Off Blend Off

		CGINCLUDE

		#include "WeatherMakerShadows.cginc"
		#pragma target 3.0
		#pragma multi_compile __ BLUR7 BLUR17

		ENDCG

        Pass
		{
            CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag            
 
            struct appdata_t
			{
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };
 
            struct v2f
			{
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };
 
            v2f vert (appdata_t v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = AdjustFullScreenUV(v.uv);
                return o;
            }
 
            float4 frag(v2f i) : SV_Target
            {
				return tex2Dlod(_MainTex, float4(i.uv, 0.0, 0.0)).r;
            }

            ENDCG
        }

		Pass
		{
			CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag     
 
            struct appdata_t
			{
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };
 
            struct v2f
			{
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;

#if defined(BLUR7)

				float2 offsets : TEXCOORD1;

#elif defined(BLUR17)

				float4 offsets : TEXCOORD1;

#endif

            };
 
            v2f vert (appdata_t v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = AdjustFullScreenUV(v.uv);

#if defined(BLUR7)

				// take top left 3 and bottom right 3 plus center pixel average
				o.offsets = float2(_WeatherMakerSunShadowMapTexture_TexelSize.x * 0.333333, _WeatherMakerSunShadowMapTexture_TexelSize.y * 0.333333);

#elif defined(BLUR17)

				// (0.4,-1.2) , (-1.2,-0.4) , (1.2,0.4) and (-0.4,1.2).
				o.offsets = float4
				(
					_WeatherMakerSunShadowMapTexture_TexelSize.x * 0.4,
					_WeatherMakerSunShadowMapTexture_TexelSize.x * 1.2,
					_WeatherMakerSunShadowMapTexture_TexelSize.y * 0.4,
					_WeatherMakerSunShadowMapTexture_TexelSize.y * 1.2
				);

#endif

                return o;
            }
 
            float4 frag(v2f i) : SV_Target
            {
				float col = tex2Dlod(_MainTex, float4(i.uv, 0.0, 0.0)).r;

#if defined(BLUR7)

				// 7 tap approximation with 2 texture lookups
				col += tex2Dlod(_MainTex, (float4(i.uv.x - i.offsets.x, i.uv.y - i.offsets.y, 0.0, 0.0))).r;
				col += tex2Dlod(_MainTex, (float4(i.uv.x + i.offsets.x, i.uv.y + i.offsets.y, 0.0, 0.0))).r;
				col *= 0.333333; // 3 total colors to average

#elif defined(BLUR17)

				// 17 tap approximation with 4 texture lookups
				col += tex2Dlod(_MainTex, (float4(i.uv.x + i.offsets.x, i.uv.y - i.offsets.w, 0.0, 0.0))).r;
				col += tex2Dlod(_MainTex, (float4(i.uv.x - i.offsets.y, i.uv.y - i.offsets.z, 0.0, 0.0))).r;
				col += tex2Dlod(_MainTex, (float4(i.uv.x + i.offsets.y, i.uv.y + i.offsets.z, 0.0, 0.0))).r;
				col += tex2Dlod(_MainTex, (float4(i.uv.x - i.offsets.x, i.uv.y + i.offsets.w, 0.0, 0.0))).r;
				col *= 0.2; // 5 total colors to average

#endif

				return col;
            }

            ENDCG
		}
    }
    Fallback Off
}
