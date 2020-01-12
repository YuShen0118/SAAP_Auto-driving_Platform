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

Shader "WeatherMaker/WeatherMakerAdditiveSoftTintable"
{
	Properties
	{
		_MainTex("Particle Texture", 2D) = "white" {}
		_InvFade("Soft Particles Factor", Range(0.01,3.0)) = 1.0
		_TintColor("Tint Color (RGBA)", color) = (1, 1, 1, 1)
		_Intensity("Intensity (float)", Range(0.01, 4.0)) = 1.0
	}

	SubShader
	{
		Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
		Blend One OneMinusSrcColor
		Cull Off
		Lighting Off
		ZWrite Off
		ColorMask RGB

		CGINCLUDE

		#include "WeatherMakerShader.cginc"

		#pragma target 3.0
		#pragma vertex vert
		#pragma fragment frag
		#pragma fragmentoption ARB_precision_hint_fastest
		#pragma glsl_no_auto_normalization
		#pragma multi_compile_particles

		struct appdata_t
		{
			float4 vertex : POSITION;
			fixed4 color : COLOR;
			float2 texcoord : TEXCOORD0;
		};

		struct v2f
		{
			float4 vertex : SV_POSITION;
			fixed4 color : COLOR;
			float2 texcoord : TEXCOORD0;
			// UNITY_FOG_COORDS(1)

#if defined(SOFTPARTICLES_ON)

			float4 projPos : TEXCOORD2;

#endif

		};

		ENDCG

		PASS
		{
			Name "MainPass"
			LOD 100

			CGPROGRAM

			v2f vert(appdata_t v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);

#if defined(SOFTPARTICLES_ON)

				o.projPos = ComputeScreenPos(o.vertex);
				COMPUTE_EYEDEPTH(o.projPos.z);

#endif

				o.color = v.color * _TintColor * _Intensity;
				o.texcoord = TRANSFORM_TEX(v.texcoord, _MainTex);

				// UNITY_TRANSFER_FOG(o, o.vertex);

				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{

#if defined(SOFTPARTICLES_ON)

				float sceneZ = LinearEyeDepth(WM_SAMPLE_DEPTH_PROJ(i.projPos));
				float partZ = i.projPos.z;
				float fade = saturate(_InvFade * (sceneZ - partZ));
				i.color.rgb = i.color.rgb * fade;

#endif

				// UNITY_APPLY_FOG_COLOR(i.fogCoord, col, fixed4(0,0,0,0)); // fog towards black due to our blend mode
				return i.color * tex2D(_MainTex, i.texcoord);
			}

			ENDCG
		}
	}
}