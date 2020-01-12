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

// receive shadows: http://www.gamasutra.com/blogs/JoeyFladderak/20140416/215612/Let_there_be_shadow.php

#include "WeatherMakerShader.cginc"

#if defined(ENABLE_FOG_LIGHTS_WITH_SHADOWS)

#include "WeatherMakerShadows.cginc"

#endif

#define FOG_LIGHT_POINT_SAMPLE_COUNT 5.0
#define FOG_LIGHT_POINT_SAMPLE_COUNT_INVERSE (1.0 / FOG_LIGHT_POINT_SAMPLE_COUNT)
#define FOG_LIGHT_SPOT_SAMPLE_COUNT 40.0
#define FOG_LIGHT_SPOT_SAMPLE_COUNT_INVERSE (1.0 / FOG_LIGHT_SPOT_SAMPLE_COUNT)

#if defined(SHADER_API_D3D9)
#define FOG_MAX_NULL_ZONE_COUNT 2
#else
#define FOG_MAX_NULL_ZONE_COUNT 8 // should match MaximumNullFogZones in WeatherMakerLightManagerScript
#endif

uniform sampler2D _FogNoise;
uniform sampler2D _FogNoiseMask;
uniform sampler3D _WeatherMakerNoiseTexture3D;

uniform float _WeatherMakerVolumetricPointSpotMultiplier = 1.0;

uniform fixed4 _FogColor;
uniform fixed4 _FogEmissionColor;
uniform fixed _FogDitherLevel;
uniform float _FogNoiseScale;
uniform float _FogNoiseAdder;
uniform float _FogNoiseMultiplier;
uniform float _FogNoiseSampleCount;
uniform float _FogNoiseSampleCountInverse;
uniform float3 _FogNoiseVelocity;
uniform float _FogHeight;
uniform float3 _FogBoxMin;
uniform float3 _FogBoxMax;
uniform float3 _FogBoxMinDir;
uniform float3 _FogBoxMaxDir;
uniform float _FogBoxDiameter;
uniform float4 _FogSpherePosition;
uniform float _MaxFogFactor;

uniform fixed _FogNoiseMaskScale;
uniform fixed2 _FogNoiseMaskOffset;
uniform fixed2 _FogNoiseMaskVelocity;
uniform fixed _FogNoiseMaskRotationSin;
uniform fixed _FogNoiseMaskRotationCos;
uniform fixed _FogCover;
uniform fixed _FogDensity;
uniform fixed _FogDensityScatter;
uniform fixed _FogLightAbsorption;
uniform fixed _FogSharpness;
uniform fixed _FogWhispiness;
uniform fixed _FogWhispinessChangeFactor;
uniform fixed _FogShadowThreshold;
uniform fixed _FogShadowMultiplier;
uniform fixed _FogShadowPower;

uniform float _FogNullZoneCount;
uniform float3 _FogNullZonesMin[FOG_MAX_NULL_ZONE_COUNT];
uniform float3 _FogNullZonesMax[FOG_MAX_NULL_ZONE_COUNT];

uniform fixed4 _FogSunShaftsParam1;
uniform fixed4 _FogSunShaftsParam2;
uniform fixed4 _FogSunShaftsTintColor;
uniform fixed4 _FogSunShaftsDitherMagic;

uniform float _FogLightShadowSampleCount;
uniform float _FogLightShadowInvSampleCount;
uniform float _FogLightShadowMaxRayLength;
uniform float _FogLightShadowMultiplier;
uniform float _FogLightShadowBrightness;
uniform float _FogShadowLightPower;
uniform float _FogShadowDecay;
uniform float _FogLightShadowDither;
uniform float4 _FogLightShadowDitherMagic;

inline float CalculateFogFactor(float depth)
{

#if defined(WEATHER_MAKER_FOG_CONSTANT)

	float fogFactor = _FogDensity * ceil(saturate(depth));

#elif defined(WEATHER_MAKER_FOG_LINEAR)

	float fogFactor = min(1.0, (depth / _ProjectionParams.z) * (_FogDensity * _ProjectionParams.z * 0.1));

#elif defined(WEATHER_MAKER_FOG_EXPONENTIAL)

	// simple height formula
	// const float extinction = 0.01;
	// float fogFactor = saturate((_FogDensity * exp(-(_WorldSpaceCameraPos.y - _FogHeight) * extinction) * (1.0 - exp(-depth * rayDir.y * extinction))) / rayDir.y);

	float fogFactor = 1.0 - saturate(1.0 / (exp(depth * _FogDensity)));

#elif defined(WEATHER_MAKER_FOG_EXPONENTIAL_SQUARED)

	float expFog = exp(depth * _FogDensity);
	float fogFactor = 1.0 - saturate(1.0 / (expFog * expFog));

#else

	float fogFactor = 0.0;

#endif

	return fogFactor;
}

inline float CalculateFogDirectionalLightScatter(float3 rayDir, float3 lightDir, float fogFactor, float depth01, float power, float multiplier)
{
	float cosAngle = max(0.0, dot(lightDir, rayDir));
	float scatter = pow(cosAngle, power);

	// scatter is reduced by depth01 value
	return scatter * GetMieScattering(cosAngle) * _WeatherMakerFogDirectionalLightScatterIntensity * depth01 * multiplier * max(0.0, 1.0 - (_FogDensity * 1.2));
}

// f = density (0 to 1)
fixed3 ComputeCloudLighting(fixed f, fixed fogDensity, float3 ray, float3 worldPos, fixed4 sunColor, out fixed alpha)
{
	fixed invFogDensity = 1.0 - fogDensity;

	// reduce scatter as overall density increases
	fixed scatterMultiplier = pow(invFogDensity, 2.0);

	// alpha approaches 1 as density increases
	alpha = pow(f, invFogDensity * pow(invFogDensity, 0.5));

	// sunlight scatter - direct line of sight
	fixed sunLightDot = max(0.0, dot(_WeatherMakerSunDirectionUp, ray));
	sunLightDot = pow(sunLightDot, _WeatherMakerSunLightPower.x);
	fixed scatterPowerDirect = (sunLightDot * sunColor.a * scatterMultiplier);
	// increase scatter power direct as f decreases
	scatterPowerDirect /= pow(f, 0.5);

	// sunglight scatter - indirect
	fixed scatterPowerIndirect = scatterMultiplier * _WeatherMakerSunColor.a * (1.0 - sunLightDot) * (1.0 - pow(f, 1.3));

	fixed lightMultiplier = sunColor.a + scatterPowerDirect + scatterPowerIndirect;
	fixed3 litFog = sunColor.rgb * lightMultiplier * _WeatherMakerSunLightPower.y;
	static const fixed fogLightAbsorptionFogDensityMultiplier =

#if defined(UNITY_COLORSPACE_GAMMA)

		4.0;

#else

		1.0;

#endif

	fixed sunLightDensityFactor = 1.0 - pow(f, _FogLightAbsorption * 100.0 * (1.0 + (_FogDensity * fogLightAbsorptionFogDensityMultiplier)));

	// reduce light by density factor
	litFog *= sunLightDensityFactor;

	// increase alpha as density factor decreases
	alpha = min(1.0, alpha / pow(sunLightDensityFactor, 0.75));

	// moonlight
	for (int i = 0; i < _WeatherMakerMoonCount; i++)
	{
		fixed moonLightDot = max(0.0, dot(_WeatherMakerMoonDirectionUp[i], ray));
		moonLightDot = pow(moonLightDot, _WeatherMakerMoonLightPower[i].x);
		scatterPowerDirect = (moonLightDot * pow(_WeatherMakerMoonLightColor[i].a, 0.75) * scatterMultiplier);
		lightMultiplier = _WeatherMakerMoonLightColor[i].a + scatterPowerDirect;
		litFog += _WeatherMakerMoonLightColor[i].rgb * lightMultiplier * _WeatherMakerMoonLightPower[i].y * sunLightDensityFactor;
	}

	// additional lights, proportional to inverse fog thickness
	litFog += ((1.0 - f) * CalculateVertexColorWorldSpace(worldPos, false).rgb);

	// ambient, proportional to inverse fog thickness
	litFog += ((1.0 - f) * _WeatherMakerAmbientLight.rgb * _AmbientLightMultiplier);

	alpha = min(1.0, pow(alpha + 0.01, 1.8));

	return litFog;
}

fixed3 ComputeFogSunShaftColor(float2 screenUV, fixed fogFactor)
{
	fixed3 color = 0.0;

	// adjust UV coordinate if needed
	float2 viewportPos = AdjustFullScreenUV(_WeatherMakerSunViewportPosition.xy);

	// determine how much to march each step - using the spread parameter (_FogSunShaftsParam1.x) we can change the length of the sun-shafts
	float2 uvMarch = (viewportPos - screenUV) * _FogSunShaftsParam1.x;

	float dither = frac(cos(dot(screenUV, ditherMagic.xy)) * ditherMagic.z);
	dither *= (frac(dot(_FogSunShaftsDitherMagic.xy, screenUV * _FogSunShaftsDitherMagic.zw)) - 0.5);

	// adjust uv direction by dither
	uvMarch *= (1.0 + (_FogSunShaftsParam2.z * dither));

	// start off with full step multiplier
	fixed stepMultiplier = _FogSunShaftsParam2.x;

	// ray march sample count times
	for (int i = 0; i < int(_FogSunShaftsParam1.y); i++)
	{
		// march from center of sun pixel towards target pixel
		screenUV += uvMarch;

		// read camera target and sum up all colors and average them
		fixed3 rgb = tex2Dlod(_MainTex2, float4(screenUV.xy, 0, 0)).rgb;

		// apply color using step multiplier and multiply by inverse sample count * weight
		color += (rgb * stepMultiplier * _FogSunShaftsParam1.w);

		// decrease step multiplier by decay
		stepMultiplier *= _FogSunShaftsParam2.y;
	}

	// multiply final color by brightness multiplier and reduce by fog factor
	return color * _FogSunShaftsTintColor.rgb * _FogSunShaftsParam1.z * (1.0 - fogFactor);
}

void ComputeDirectionalLightFog(float3 rayOrigin, float3 rayDir, float rayLength, float fogFactor, float depth01, out fixed3 lightColor, float2 screenUV)
{
	// add full ambient as sun intensity approaches 0
	fixed3 ambient = min(1.0, _WeatherMakerAmbientLight.rgb * _AmbientLightMultiplier * max(0.0, 1.0 - _WeatherMakerSunColor.a));
	lightColor = ambient * fogFactor;
	// sun light + scatter
	fixed3 sunLightColor = (_WeatherMakerSunColor.rgb * _WeatherMakerSunColor.a * _DirectionalLightMultiplier);
	float scatter = CalculateFogDirectionalLightScatter(rayDir, _WeatherMakerSunDirectionUp, fogFactor, depth01, _WeatherMakerSunLightPower.x, _FogDensityScatter);

#if defined(ENABLE_FOG_LIGHTS_WITH_SHADOWS)

	if (rayLength < _FogLightShadowMaxRayLength)
	{
		float4 wpos = float4(rayOrigin, 1.0);
		float shadowPower = 0.0;
		float viewZ = 0.0;
		float4 cascadeWeights;
		float4 samplePos;
		float shadowDepth;
		float lightDot = max(_FogShadowDecay, dot(_WeatherMakerSunDirectionUp, rayDir));

		// dithering
		float dither = (1.0 + (_FogLightShadowDither / max(0.4, fogFactor + (2 * lightDot)) * frac(cos(dot(screenUV, ditherMagic.xy)) * ditherMagic.z) *
			(frac(dot(_FogLightShadowDitherMagic.xy, screenUV * _FogLightShadowDitherMagic.zw)) - 0.5)));

		lightDot = pow(lightDot, _FogShadowLightPower);
		lightDot = 1.0 + (_FogLightShadowMultiplier * lightDot);



#if defined(SHADER_API_D3D9)

		float sampleCount = 16;
		float invSampleCount = 1.0 / 16.0;

#else

		float sampleCount = floor(min(rayLength * 3, _FogLightShadowSampleCount));
		float invSampleCount = (1.0 / (float)sampleCount);

#endif

		float stepAmount = (min(rayLength, _FogLightShadowMaxRayLength) * invSampleCount);
		float3 stepDir = rayDir * stepAmount;

		stepDir *= dither;

		// un-comment to use softer shadows by multi-sampling, not much of a difference in appearance, especially if blurring the fog
		//_WeatherMakerSunShadowMapTexture_Sample4 = float2(_WeatherMakerSunShadowMapTexture_TexelSize.x * 0.5, _WeatherMakerSunShadowMapTexture_TexelSize.x * 0.5);

		// for sun, ray march through the shadow map
		for (int i = 0; i < int(sampleCount); i++)
		{
			viewZ += stepAmount;
			wpos.xyz += stepDir;

			cascadeWeights = GET_CASCADE_WEIGHTS(wpos, viewZ);
			samplePos = GET_SHADOW_COORDINATES(wpos, cascadeWeights);

			shadowPower += UNITY_SAMPLE_SHADOW(_WeatherMakerSunShadowMapTexture, samplePos);
			//shadowPower += UNITY_SAMPLE_SHADOW_4(_WeatherMakerSunShadowMapTexture, samplePos);
		}

		// general scattering
		lightColor += ((fogFactor * (sunLightColor + (sunLightColor * scatter))) *

		// ray scattering
		(1.0 + (max(0.0, (1.0 - (2.0 * fogFactor))) * shadowPower * invSampleCount * lightDot * _FogLightShadowBrightness)));
	}
	else
	{

#endif

		lightColor += (fogFactor * (sunLightColor + (sunLightColor * scatter)));

#if defined(ENABLE_FOG_LIGHTS_WITH_SHADOWS)

	}

#endif

	// moon light + scatter
	for (int i = 0; i < _WeatherMakerMoonCount; i++)
	{
		fixed3 moonLightColor = _WeatherMakerMoonLightColor[i].rgb * _WeatherMakerMoonLightColor[i].a * _DirectionalLightMultiplier;
		scatter = CalculateFogDirectionalLightScatter(rayDir, _WeatherMakerMoonDirectionUp[i], fogFactor, depth01, _WeatherMakerMoonLightPower[i].x, _FogDensityScatter);
		lightColor += (fogFactor * (moonLightColor + (moonLightColor * scatter)));
	}
}
 
inline void ComputeLightColorForFogPointSpotLight
(
	float3 rayOrigin,
	float3 rayDir,
	float lightAmount,
	float distanceToLight,
	float isPointLight,
	float4 lightPos,
	fixed4 lightColor,
	float4 lightAtten,
	float4 lightDir,
	float4 lightEnd,
    float lightMultiplier,
	inout fixed3 accumLightColor)
{
	// amount to move for each sample
	float3 step;

	// fog factor recuder is the amount of fog in front of the light, used to reduce the light
	float fogFactorReducer = 1.0 - CalculateFogFactor(distanceToLight * 0.33);

	// amount of fog on light ray
	float fogFactorOnRay = CalculateFogFactor(lightAmount);

	// sample points along the ray
	float lightSample;
	float attenSample = 0.0;
	float eyeLightDot;
	float3 startPos = rayOrigin + (rayDir * distanceToLight);
	float3 currentPos;
	float3 toLight;

	if (isPointLight)
	{
		step = rayDir * (lightAmount * FOG_LIGHT_POINT_SAMPLE_COUNT_INVERSE);
		currentPos = startPos - (step * 0.5);
		lightSample = 0.0;

		for (int i = 0; i < int(FOG_LIGHT_POINT_SAMPLE_COUNT); i++)
		{
			currentPos += step;
			toLight = currentPos - lightPos.xyz;
			lightSample += dot(toLight, toLight);
		}

		// average samples
		lightSample *= FOG_LIGHT_POINT_SAMPLE_COUNT_INVERSE;

		// calculate atten from distance from center
		lightSample = max(0.0, 1.0 - (lightSample * lightAtten.w));
		lightSample = lightSample * lightSample * CalculateFogFactor(lightAmount) * lightColor.a;

		// as camera approaches light position, reduce amount of light
		// right next to the light there is less light travelling to the eye through the fog
		toLight = _WorldSpaceCameraPos - lightPos.xyz;
		float d = dot(toLight, toLight);
		lightAmount = lightSample * clamp(d * lightAtten.w, 0.2, 1.0);
	}
	else
	{
		float dotSample1 = 0.0;
		float dotSample2 = 0.0;
		float distanceSample = 0.0;

		lightSample = 9999999.0;
		step = rayDir * (lightAmount * FOG_LIGHT_SPOT_SAMPLE_COUNT_INVERSE);
		currentPos = startPos - (step * 0.5);
		for (int i = 0; i < int(FOG_LIGHT_SPOT_SAMPLE_COUNT); i++)
		{
			currentPos += step;
			toLight = currentPos - lightPos.xyz;
			eyeLightDot = dot(toLight, toLight);
			distanceSample += eyeLightDot;
			lightSample = min(eyeLightDot, lightSample);
			eyeLightDot = saturate(((dot(normalize(toLight), lightDir.xyz)) - lightAtten.x) * lightAtten.y);
			dotSample1 = max(eyeLightDot, dotSample1);
			dotSample2 += eyeLightDot;
		}

		// calculate dot attenuation, light at more of an angle is dimmer
		eyeLightDot = (dotSample1 * 0.75) + (dotSample2 * FOG_LIGHT_SPOT_SAMPLE_COUNT_INVERSE * 0.25);
		dotSample1 = pow(eyeLightDot, _WeatherMakerFogLightFalloff.x * lightPos.w);

		// calculate light attenuation
		lightSample = 1.0 / (1.0 + (lightSample * lightAtten.z));

		// increase as eye looks at center from forward direction
		eyeLightDot = max(0.0, dot(-rayDir, lightDir.xyz));
		dotSample1 *= (1.0 + (lightColor.a * 2.0 * eyeLightDot * eyeLightDot * eyeLightDot));

		// reduce light right near the tip to eliminate hard edges
		lightAmount = (min(1.0, distanceSample * FOG_LIGHT_SPOT_SAMPLE_COUNT_INVERSE)) * lightSample * dotSample1 * lightColor.a;
	}

	// apply color
	accumLightColor += (lightColor.rgb * lightAmount * fogFactorReducer * fogFactorOnRay * lightMultiplier

#if defined(UNITY_COLORSPACE_GAMMA)

		// brighten up gamma space to make it look more like linear
		* 1.8

#endif
		
		);
}

void ComputePointSpotLightFog(float3 rayOrigin, float3 rayDir, float rayLength, float fogFactor, inout fixed3 lightColor)
{
    float lightMultiplier = _PointSpotLightMultiplier * _WeatherMakerVolumetricPointSpotMultiplier;
	float lightAmount;
	float distanceToLight;
	int lightIndex;

	// point lights
	for (lightIndex = 0; lightIndex < _WeatherMakerPointLightCount; lightIndex++)
	{
		// get the length of the ray intersecting the point light sphere
		if (RaySphereIntersect(rayOrigin, rayDir, rayLength, _WeatherMakerPointLightPosition[lightIndex], lightAmount, distanceToLight))
		{
			// compute lighting for the point light
			ComputeLightColorForFogPointSpotLight(rayOrigin, rayDir, lightAmount, distanceToLight, 1, _WeatherMakerPointLightPosition[lightIndex],
				_WeatherMakerPointLightColor[lightIndex], _WeatherMakerPointLightAtten[lightIndex], 0, 0, lightMultiplier, lightColor);
		}
	}

	// spot lights
	for (lightIndex = 0; lightIndex < _WeatherMakerSpotLightCount; lightIndex++)
	{
		// get the length of the ray intersecting the spot light cone
		if (RayConeIntersect(rayOrigin, rayDir, rayLength, _WeatherMakerSpotLightPosition[lightIndex], _WeatherMakerSpotLightSpotDirection[lightIndex],
			_WeatherMakerSpotLightSpotEnd[lightIndex], _WeatherMakerSpotLightAtten[lightIndex], lightAmount, distanceToLight))
		{
			// compute lighting for the spot light
			ComputeLightColorForFogPointSpotLight(rayOrigin, rayDir, lightAmount, distanceToLight, 0, _WeatherMakerSpotLightPosition[lightIndex],
				_WeatherMakerSpotLightColor[lightIndex], _WeatherMakerSpotLightAtten[lightIndex], _WeatherMakerSpotLightSpotDirection[lightIndex],
				_WeatherMakerSpotLightSpotEnd[lightIndex], lightMultiplier, lightColor);
		}
	}
}

// f is fog factor, rayLength is distance of fog in ray, savedDepth is depth buffer
fixed4 ComputeFogLighting(float3 rayOrigin, float3 rayDir, float rayLength, float fogFactor, float depth01, float2 screenUV, float noise)
{
	// skip expensive lighting where there is no fog
	if (fogFactor < 0.004)
	{
		return fixed4(0.0, 0.0, 0.0, 0.0);
	}
	else
	{
		fixed4 lightColor;

		// directional light / ambient
		ComputeDirectionalLightFog(rayOrigin, rayDir, rayLength, fogFactor, depth01, lightColor.rgb, screenUV);

#if defined(ENABLE_FOG_LIGHTS) || defined(ENABLE_FOG_LIGHTS_WITH_SHADOWS)

		ComputePointSpotLightFog(rayOrigin, rayDir, rayLength, fogFactor, lightColor.rgb);

#endif

		lightColor.rgb *= _FogColor.rgb * noise;

#if defined(ENABLE_FOG_SUN_SHAFTS)

		lightColor.rgb += ComputeFogSunShaftColor(screenUV, fogFactor);

#endif

		lightColor.a = fogFactor;
		ApplyDither(lightColor.rgb, screenUV, _FogDitherLevel);
		return lightColor;
	}
}

inline float CalculateFogNoise3D(float3 pos, float3 rayDir, float rayLength, float scale, float3 velocity)
{
	float n = 0.0;
	float3 step = rayDir * scale;
	pos *= scale;
	for (int i = 0; i < int(_FogNoiseSampleCount); i++)
	{
		n += tex3Dlod(_WeatherMakerNoiseTexture3D, float4(pos + velocity, -999.0)).a;
		pos += step;
	}

	return ((n * _FogNoiseSampleCountInverse) + _FogNoiseAdder) * _FogNoiseMultiplier;
}

inline float CalculateFogNoise3DOne(float3 pos, float scale, float3 velocity)
{
	return tex3Dlod(_WeatherMakerNoiseTexture3D, float4((pos * scale) + velocity, -999.0)).a;
}

inline void RaycastFogBoxFullScreen(float3 rayDir, float3 forwardLine, inout float depth, out float3 startPos, out float noise)
{
	// depth is 0-1 value, which needs to be changed to world space distance
	startPos = _WorldSpaceCameraPos + (forwardLine * depth);

	// calculate depth exactly in world space
	depth = distance(startPos, _WorldSpaceCameraPos);
	float origDepth = depth;
	float distanceToBox;

#if defined(ENABLE_FOG_HEIGHT)

	// cast ray, get amount of intersection with the fog box
	float3 boxMin, boxMax;
	GetFullScreenBoundingBox(_FogHeight, boxMin, boxMax);
	RayBoxIntersect(_WorldSpaceCameraPos, rayDir, origDepth, boxMin, boxMax, depth, distanceToBox);

	// update world pos with the new intersect point
	startPos = _WorldSpaceCameraPos + (rayDir * distanceToBox);

#if defined(ENABLE_FOG_NOISE)

#if defined(ENABLE_FOG_NOISE_HEIGHT)

	if (distanceToBox > 0.5)
	{
		// re-cast for a new box to vary the height
		float3 recast = _WorldSpaceCameraPos + (rayDir * max(distanceToBox, depth));
		float shapeNoise = (CalculateFogNoise3DOne(recast, _FogNoiseScale * 0.25, _FogNoiseVelocity) - 0.5) * 0.5;
		GetFullScreenBoundingBox(_FogHeight * (1.0 + shapeNoise), boxMin, boxMax);
		RayBoxIntersect(_WorldSpaceCameraPos, rayDir, origDepth, boxMin, boxMax, depth, distanceToBox);
		startPos = _WorldSpaceCameraPos + (rayDir * distanceToBox);
	}

#endif

	// calculate noise
	noise = CalculateFogNoise3D(startPos, rayDir, origDepth, _FogNoiseScale, _FogNoiseVelocity);

	// remove noise where there is no fog
	noise *= (depth > 0.0 && _FogDensity > 0.0);

#else

	noise = 1.0;

#endif

#elif defined(ENABLE_FOG_NOISE)

	startPos = _WorldSpaceCameraPos;
	noise = CalculateFogNoise3D(startPos, rayDir, depth, _FogNoiseScale, _FogNoiseVelocity);

#else

	startPos = _WorldSpaceCameraPos;
	noise = 1.0;

#endif

#if defined(WEATHER_MAKER_FOG_ENABLE_NULL_FOG_ZONES) && SHADER_TARGET >= 30

	float nullDepth;
	for (int i = 0; i < int(_FogNullZoneCount); i++)
	{
		if (RayBoxIntersect(_WorldSpaceCameraPos, rayDir, origDepth, _FogNullZonesMin[i], _FogNullZonesMax[i], nullDepth, distanceToBox))
		{
			startPos += (rayDir * (distanceToBox + nullDepth));
			depth -= nullDepth;
		}
	}

#endif

}

// returns the original scene depth
inline void RaycastFogBox(float3 rayDir, float3 normal, inout float depth, out float3 startPos, out float noise)
{	
	// cast ray, get amount of intersection with the fog box
	float origDepth = depth;
	float distanceToBox;
	RayBoxIntersect(_WorldSpaceCameraPos, rayDir, origDepth, _FogBoxMin, _FogBoxMax, depth, distanceToBox);
	float falloff = depth / _FogBoxDiameter;
	depth *= pow(falloff, 0.5);
	startPos = _WorldSpaceCameraPos + (rayDir * distanceToBox);

#if defined(ENABLE_FOG_NOISE)

	// calculate noise
	noise = CalculateFogNoise3D(startPos, rayDir, depth, _FogNoiseScale, _FogNoiseVelocity);

#else

	noise = 1.0;

#endif

	// reset startPos to new point
	startPos = _WorldSpaceCameraPos + (rayDir * distanceToBox);
}

// returns the original scene depth
inline float RaycastFogSphere(float3 rayDir, float3 normal, inout float depth, out float3 startPos, out float noise)
{
	float savedDepth = depth;
	float distanceToSphere;
	float4 pos = _FogSpherePosition;
	RaySphereIntersect(_WorldSpaceCameraPos, rayDir, savedDepth, pos, depth, distanceToSphere);
	float radius = (depth * 0.5);
	float falloff = ((radius * radius) / _FogSpherePosition.w);
	depth *= pow(falloff, 0.5);

	startPos = _WorldSpaceCameraPos + (rayDir * distanceToSphere);

#if defined(ENABLE_FOG_NOISE)

	// calculate noise
	noise = CalculateFogNoise3D(startPos, rayDir, depth, _FogNoiseScale, _FogNoiseVelocity);

#else

	noise = 1.0;

#endif

	startPos = _WorldSpaceCameraPos + (rayDir * distanceToSphere);
	return savedDepth;
}

// sphere is xyz, w = radius squared, returns clarity
inline float RayMarchFogSphere(volumetric_data i, int iterations, float4 sphere, float density, float outerDensity, out float clarity, out float3 rayDir, out float3 sphereCenterViewSpace, out float maxDistance)
{
	float2 screenUV = i.projPos.xy / i.projPos.w;
	maxDistance = length(DECODE_EYEDEPTH(WM_SAMPLE_DEPTH(screenUV)) / normalize(i.viewPos).z);
	//float depthBufferDepth = LinearEyeDepth(WM_SAMPLE_DEPTH_PROJ(i.projPos));
	rayDir = normalize(i.viewPos.xyz);
	sphereCenterViewSpace = mul((float3x3)UNITY_MATRIX_V, (_WorldSpaceCameraPos - sphere.xyz));
	float invSphereRadiusSquared = 1.0 / sphere.w;

	// calculate sphere intersection
	float b = -dot(rayDir, sphereCenterViewSpace);
	float c = dot(sphereCenterViewSpace, sphereCenterViewSpace) - sphere.w;
	float d = sqrt((b * b) - c);
	float dist = b - d;
	float dist2 = b + d;

	/*
	float fA = dot(rayDir, rayDir);
	float fB = 2 * dot(rayDir, sphereCenterViewSpace);
	float fC = dot(sphereCenterViewSpace, sphereCenterViewSpace) - sphere.w;
	float fD = fB * fB - 4 * fA * fC;
	// if (fD <= 0.0f) { return; } // not sure if this is needed, doesn't seem to trigger very often
	float recpTwoA = 0.5 / fA;
	float DSqrt = sqrt(fD);
	// the distance to the front of sphere, or 0 if inside the sphere. This is the distance from the camera where sampling begins.
	float dist = max((-fB - DSqrt) * recpTwoA, 0);
	// total distance to the back of the sphere.
	float dist2 = max((-fB + DSqrt) * recpTwoA, 0);
	*/

	// stop at the back of the sphere or depth buffer, whichever is the smaller distance.
	float backDepth = min(maxDistance, dist2);

	// calculate initial sample distance, and the distance between samples.
	float samp = dist;
	float step_distance = (backDepth - dist) / (float)iterations;

	// how much does each step get modified? approaches 1 with distance.
	float step_contribution = (1 - 1 / pow(2, step_distance)) * density;

	// 1 means no fog, 0 means completely opaque fog
	clarity = 1;

	for (int i = 0; i < iterations; i++)
	{
		float3 position = sphereCenterViewSpace + (rayDir * samp);
		float val = saturate(outerDensity * (1.0 - (dot(position, position) * invSphereRadiusSquared)));
		clarity *= (1.0 - saturate(val * step_contribution));
		samp += step_distance;
	}

	return clarity;
}

// VERTEX AND FRAGMENT SHADERS ----------------------------------------------------------------------------------------------------

volumetric_data fog_volume_vertex_shader(vertex_uv_normal v)
{
	return GetVolumetricData(v.vertex, v.normal);
}

fixed4 fog_box_full_screen_fragment_shader(full_screen_fragment i) : SV_TARGET
{

#if defined(FOG_NONE)

	return fixed4(0.0, 0.0, 0.0, 0.0);

#else

	float depth01 = GetDepth01(i.uv.xy);
	float noise;
	float3 rayDir = normalize(i.forwardLine);
	float3 startPos;
	float depth = depth01; // gets set to the fog amount on the ray
	RaycastFogBoxFullScreen(rayDir, i.forwardLine, depth, startPos, noise);
	float fogFactor = saturate(CalculateFogFactor(depth) * noise);
	return ComputeFogLighting(startPos, rayDir, depth, fogFactor, depth01, i.uv, noise);

#endif

}

inline void PreFogFragment(inout volumetric_data i, out float depth, out float depth01, out float2 screenUV)
{
	// get the depth of this pixel
	screenUV = i.projPos.xy / i.projPos.w;
	i.rayDir = normalize(i.rayDir);
	float depthBufferValue = WM_SAMPLE_DEPTH(screenUV);
	depth01 = Linear01Depth(depthBufferValue);
	depth = length(DECODE_EYEDEPTH(depthBufferValue) / normalize(i.viewPos).z);
}

inline fixed4 PostFogFragment(float3 startPos, float3 rayDir, float amount, float depth01, float noise, float2 screenUV)
{
	float fogFactor = saturate(CalculateFogFactor(amount) * noise);
	return ComputeFogLighting(startPos, rayDir, amount, fogFactor, depth01, screenUV, noise);
}

fixed4 fog_box_fragment_shader(volumetric_data i) : SV_TARGET
{

#if defined(FOG_NONE)

	return fixed4(0, 0, 0, 0);

#else

	float noise;
	float2 screenUV;
	float depth, depth01;
	PreFogFragment(i, depth, depth01, screenUV);
	float3 startPos;
	RaycastFogBox(i.rayDir, i.normal, depth, startPos, noise);
	return PostFogFragment(startPos, i.rayDir, depth, depth01, noise, screenUV);

#endif

}

fixed4 fog_sphere_fragment_shader(volumetric_data i) : SV_TARGET
{

#if defined(FOG_NONE)

	return fixed4(0, 0, 0, 0);

#else

	float noise;
	float2 screenUV;
	float3 startPos;
	float depth, depth01;
	PreFogFragment(i, depth, depth01, screenUV);
	RaycastFogSphere(i.rayDir, i.normal, depth, startPos, noise);
	return PostFogFragment(startPos, i.rayDir, depth, depth01, noise, screenUV);

#endif

}
