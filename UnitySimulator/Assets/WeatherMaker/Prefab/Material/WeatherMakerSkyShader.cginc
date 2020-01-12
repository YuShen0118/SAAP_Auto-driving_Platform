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

#include "WeatherMakerFogShader.cginc"

// reduce scale of cloud noise
#define SCALE_REDUCER 0.1

// reduces night color by cloud alpha * this value
#define CLOUD_ALPHA_NIGHT_COLOR_REDUCER_MULTIPLIER 2.0

#ifndef SKYBOX_COLOR_IN_TARGET_COLOR_SPACE
#if defined(SHADER_API_MOBILE)
#define SKYBOX_COLOR_IN_TARGET_COLOR_SPACE 1
#else
#define SKYBOX_COLOR_IN_TARGET_COLOR_SPACE 0
#endif
#endif

struct procedural_sky_info
{
	fixed3 inScatter;
	fixed3 outScatter;
	fixed4 skyColor;
};

struct appdata
{
	float4 vertex : POSITION;
	float2 uv : TEXCOORD0;
};

struct v2fSky
{
	float4 vertex : SV_POSITION;
	float2 uv : TEXCOORD0;
	float3 ray : NORMAL;
	fixed3 inScatter : COLOR0;
	fixed3 outScatter : COLOR1;
};

struct v2fShadow
{
	float4 vertex : SV_POSITION;
	float3 worldPos : TEXCOORD0;
};

sampler2D _DawnDuskTex;
float4 _DawnDuskTex_ST;
sampler2D _NightTex;
float4 _NightTex_ST;
fixed _DayMultiplier;
fixed _DawnDuskMultiplier;
fixed _NightMultiplier;
fixed _NightSkyMultiplier;
fixed _NightVisibilityThreshold;
fixed _NightIntensity;
fixed _NightTwinkleSpeed;
fixed _NightTwinkleVariance;
fixed _NightTwinkleMinimum;
fixed _NightTwinkleRandomness;


fixed3 _SkyTintColor;
fixed3 _WeatherMakerSkyGradientColor = fixed3(1.0, 1.0, 1.0);
fixed _WeatherMakerSkyDitherLevel;
fixed _WeatherMakerSkySamples = 2.0;
fixed _WeatherMakerSkyMieG;
fixed _WeatherMakerSkyMieG2;
fixed _WeatherMakerSkyAtmosphereThickness = 1.0;
fixed4 _WeatherMakerSkyRadius; // outer, outer * outer, inner, inner * inner
fixed4 _WeatherMakerSkyMie; // x, y, z, w
fixed4 _WeatherMakerSkyLightScattering;
fixed4 _WeatherMakerSkyLightPIScattering;
fixed3 _WeatherMakerSkyTintColor;
fixed4 _WeatherMakerSkyScale; // scale factor, scale depth, scale / scale depth, camera height

// brings clouds down at the horizon at the cost of stretching them over the top
fixed _WeatherMakerCloudRayOffset;

inline fixed GetMiePhase(fixed size, fixed eyeCos, fixed eyeCos2, fixed power)
{
	fixed temp = 1.0 + _WeatherMakerSkyMieG2 + (2 * _WeatherMakerSkyMieG * eyeCos);
	temp = max(1.0e-4, smoothstep(0.0, 0.005, temp) * temp);
	fixed mie = saturate(size * _WeatherMakerSkyMie.x * ((1.0 + eyeCos2) / temp));
	return pow(mie, power);
}

inline fixed GetSkyMiePhase(fixed eyeCos, fixed eyeCos2)
{
	return _WeatherMakerSkyMie.x * (1.0 + eyeCos2) / pow(_WeatherMakerSkyMie.y + _WeatherMakerSkyMie.z * eyeCos, 1.5);
}

inline fixed GetRayleighPhase(fixed eyeCos2)
{
	return 0.75 + 0.75 * eyeCos2;
}

inline fixed GetRayleighPhase(fixed3 light, fixed3 ray)
{
	fixed eyeCos = dot(light, ray);
	return GetRayleighPhase(eyeCos * eyeCos);
}

inline fixed CalcSunSpot(fixed size, fixed3 vec1, fixed3 vec2)
{
	half3 delta = vec1 - vec2;
	half dist = length(delta);
	half spot = 1.0 - smoothstep(0.0, size, dist);
	return saturate(80000.0 * spot * spot);
}

inline fixed4 GetSunColorHighQuality(float3 sunNormal, fixed4 sunColor, fixed size, float3 ray)
{
	fixed eyeCos = -dot(sunNormal, ray);
	fixed eyeCos2 = eyeCos * eyeCos;
	fixed mie = GetMiePhase(size, eyeCos, eyeCos2, 1.0);
	return (mie * sunColor);
}

inline fixed4 GetSunColorFast(float3 sunNormal, fixed4 sunColor, fixed size, float3 ray)
{
	fixed sun = CalcSunSpot(size, sunNormal, ray);
	return (sun * sunColor);
}

inline float GetSkyScale(float inCos)
{
	float x = 1.0 - inCos;
#if defined(SHADER_API_N3DS)
	// The polynomial expansion here generates too many swizzle instructions for the 3DS vertex assembler
	// Approximate by removing x^1 and x^2
	return 0.25 * exp(-0.00287 + x * x * x * (-6.80 + x * 5.25));
#else
	return 0.25 * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
#endif

}

procedural_sky_info CalculateScatteringCoefficients(float3 lightDir, fixed3 lightColor, float scale, float3 eyeRay)
{
	procedural_sky_info o;
	eyeRay.y = max(-0.01, eyeRay.y);

	float outerRadius = _WeatherMakerSkyRadius.x;
	float outerRadius2 = _WeatherMakerSkyRadius.y;
	float innerRadius = _WeatherMakerSkyRadius.z;
	float innerRadius2 = _WeatherMakerSkyRadius.w;
	float scaleFactor = _WeatherMakerSkyScale.x * scale;
	float scaleDepth = _WeatherMakerSkyScale.y;
	float scaleFactorOverDepth = _WeatherMakerSkyScale.z;
	float cameraHeight = _WeatherMakerSkyScale.w;

	// the following is copied from Unity procedural sky shader
	float3 cameraPosition = float3(0.0, innerRadius + cameraHeight, 0.0);
	float far = sqrt(outerRadius2 + innerRadius2 * eyeRay.y * eyeRay.y - innerRadius2) - innerRadius * eyeRay.y;
	float startDepth = exp(scaleFactorOverDepth * (-cameraHeight));
	float startAngle = dot(eyeRay, cameraPosition) / (innerRadius + cameraHeight);
	float startOffset = startDepth * GetSkyScale(startAngle);
	float sampleLength = far / _WeatherMakerSkySamples;
	float scaledLength = sampleLength * scaleFactor;
	float3 sampleRay = eyeRay * sampleLength;
	float3 samplePoint = cameraPosition + sampleRay * 0.5;
	float3 color = float3(0.0, 0.0, 0.0);

	// Loop through the sample rays
	for (int i = 0; i < int(_WeatherMakerSkySamples); i++)
	{
		float height = length(samplePoint);
		float invHeight = 1.0 / height;
		float depth = exp(scaleFactorOverDepth * (innerRadius - height));
		float scaleAtten = depth * scaledLength;
		float eyeAngle = dot(eyeRay, samplePoint) * invHeight;
		float lightAngle = dot(lightDir, samplePoint) * invHeight;
		float lightScatter = startOffset + depth * (GetSkyScale(lightAngle) - GetSkyScale(eyeAngle));
		float3 lightAtten = exp(-lightScatter * (_WeatherMakerSkyLightPIScattering.xyz + _WeatherMakerSkyLightPIScattering.w));
		color += (lightAtten * scaleAtten);
		samplePoint += sampleRay;
	}

	o.inScatter = lightColor * color * _WeatherMakerSkyLightScattering.xyz;
	o.outScatter = lightColor * color * _WeatherMakerSkyLightScattering.w;

	return o;
}

procedural_sky_info CalculateScatteringColor(float3 lightDir, fixed3 lightColor, fixed sunSize, float3 eyeRay, fixed3 inScatter, fixed3 outScatter)
{
	float eyeCos = dot(lightDir, eyeRay);
	float eyeCos2 = eyeCos * eyeCos;
	procedural_sky_info o;
	o.inScatter = inScatter;
	o.outScatter = outScatter;

	o.skyColor.rgb = GetRayleighPhase(eyeCos2) * inScatter;
	o.skyColor.rgb += (outScatter * GetSkyMiePhase(eyeCos, eyeCos2));
	
	// draws the sun, not used for now
	//o.skyColor.rgb += GetMiePhase(sunSize, eyeCos, eyeCos2, 1.0) * outScatter;
	o.skyColor.a = max(o.skyColor.r, max(o.skyColor.g, o.skyColor.b));

#if defined(UNITY_COLORSPACE_GAMMA) && SKYBOX_COLOR_IN_TARGET_COLOR_SPACE
	o.skyColor.rgb = sqrt(o.skyColor.rgb);
#endif

	return o;
}

fixed3 GetNightColor(float3 ray, float2 uv)
{
	fixed3 nightColor = tex2D(_NightTex, uv).rgb * _NightIntensity;
	nightColor *= (nightColor >= _NightVisibilityThreshold);
	fixed maxValue = max(nightColor.r, max(nightColor.g, nightColor.b));

#if defined(ENABLE_NIGHT_TWINKLE)

	fixed twinkleRandom = _NightTwinkleRandomness * RandomFloat(ray * _WeatherMakerTime.y);
	fixed twinkle = (maxValue > _NightTwinkleMinimum) * (twinkleRandom + (_NightTwinkleVariance * sin(_NightTwinkleSpeed * _WeatherMakerTime.y * maxValue)));
	nightColor *= (1.0 + twinkle);

#endif

	nightColor *= _NightSkyMultiplier;

	return nightColor * _NightIntensity * _NightSkyMultiplier;
}

#if defined(ENABLE_CLOUDS) || defined(ENABLE_CLOUDS_MASK)

inline float CloudNoise(sampler2D tex, float2 xz, float cover, float2 vel)
{
	return CalculateNoiseXZ(tex, float3(xz.x, 0.0, xz.y), _FogNoiseScale, 0.0, vel, _FogNoiseMultiplier, cover);
}

inline float CloudFBM(float2 pos, float cover)
{
	float f1 = (CloudNoise(_FogNoise, pos, cover, _FogNoiseVelocity) * (1.0 - _FogWhispiness));
	float f2 = (CloudNoise(_FogNoise, pos, cover, _FogNoiseVelocity * (1.0 - _FogWhispinessChangeFactor)) * _FogWhispiness);
	return (f1 + f2) * _FogNoiseMultiplier;
}

fixed ComputeCloudFBMFromXZ(float2 xz)
{
	// calculate cloud values
	xz = xz * SCALE_REDUCER;
	fixed c = lerp(0.15, 0.5, _FogCover);
	fixed cloudDensity = CloudFBM(xz, c);
	cloudDensity = saturate(1.0 - (pow(_FogSharpness, cloudDensity - (1.0 - c))));

#if defined(ENABLE_CLOUDS_MASK)

	float2 maskRotated = RotateUV(xz, _FogNoiseMaskRotationSin, _FogNoiseMaskRotationCos);
	float maskNoise = CalculateNoiseXZ(_FogNoiseMask, float3(maskRotated.x, 0.0, maskRotated.y), _FogNoiseMaskScale, _FogNoiseMaskOffset, _FogNoiseMaskVelocity, 1.0, 0.0);
	cloudDensity *= maskNoise;

#endif

	return cloudDensity;
}

float3 CloudRaycastWorldPos(float3 ray)
{
	float3 normal = float3(0.0, 1.0, 0.0);
	float denom = dot(normal, ray);

	// get base plane intersection
	if (denom < 0.00001)
	{
		// early exit, don't draw the bottom half of the clouds - performance gain of not calculating these pixels with the branch is greater than the below processing code
		// the fragment shader will render the bottom half in batches and will branch the same way in all of the GPU processors
		return float3(0.0, 0.0, 0.0);
	}

	float3 pos = float3(0, _FogHeight, 0);
	float3 cameraPos = float3(_WorldSpaceCameraPos.x, 0.0, _WorldSpaceCameraPos.z);
	float t = dot(pos, normal) / denom;
	return cameraPos + (ray * t);
}

fixed ComputeCloudFBM(float3 ray, out float3 worldPos)
{
	worldPos = CloudRaycastWorldPos(ray);
	return ComputeCloudFBMFromXZ(worldPos.xz);
}

fixed4 ComputeCloudColor(float3 ray, fixed4 sunColor, out float3 worldPos)
{
	fixed f = ComputeCloudFBM(ray, worldPos);
	if (f < 0.005)
	{
		// fast out for transparent areas, avoids a lot of math and calculations
		return fixed4(0.0, 0.0, 0.0, 0.0);
	}

	// compute lighting
	fixed alpha;
	fixed density = min(1.0, _FogDensity * 1.5);
	fixed3 litCloud = ComputeCloudLighting(f, density, ray, worldPos, sunColor, alpha);

	// apply additional variance to color based on density
	fixed densityColorMultiplier = max(f, 1.0 - _FogDensity);

	return fixed4((_FogColor * _WeatherMakerSkyGradientColor * fixed4(litCloud, 1.0) * densityColorMultiplier) + (_FogEmissionColor.rgb * _FogEmissionColor.a), alpha);
}

#endif
