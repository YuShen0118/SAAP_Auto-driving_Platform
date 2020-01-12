Weather Maker, (c) 2016 Digital Ruby, LLC
http://www.digitalruby.com
Created by Jeff Johnson

*** A NOTE ABOUT PIRACY ***

If you got this asset off of leak forums or any other horrible evil pirate site, please consider buying it from the Unity asset store at https://www.assetstore.unity3d.com/en/#!/content/60955?aid=1011lGnL. This asset is only legally available from the Unity Asset Store.

I'm a single indie dev supporting my family by spending hundreds and thousands of hours on this and other assets. It's very offensive, rude and just plain evil to steal when I (and many others) put so much hard work into the software.

Thank you.

*** END NOTE ABOUT PIRACY ***

Current Version : 2.9.1

For Change Log, See ChangeLog.txt.

Welcome to Weather Maker! I'm Jeff Johnson, and I've spent hundreds of hours on this asset. Please send me your suggestions, feedback and bug reports to support@digitalruby.com.

What is Weather Maker? Weather Maker is a complete sky and weather system for Unity. Clouds, precipitation, lighting, storms, wind, sound effects and more are built in.

Setup Instructions:
- Install latest supported Unity version: Unity 5.6.2 or newer.
- Delete Rain Maker from your project if it is loaded. It is not compatible in the same project with Weather Maker, and is not needed anyway.
- If you want to use the demo scenes, import standard assets -> environment and the Unity Post Processing stack at https://www.assetstore.unity3d.com/en/#!/content/83912?aid=1011lGnL.
- Drag ./WeatherMaker/Prefab/Prefab/WeatherMakerPrefab (or for a 2D game, WeatherMakerPrefab2D) into your scene at the root of the hierarchy.
- For 3D, if you will be using lightning, set your main camera far plane to 2000+.
- For 3D, see the Suns and Moons array on WeatherScript on the root of the refab. Configure them how you like. Moons are setup to have Earths moon and Mars by default.
- Weather Maker is currently set up to work in a single first person camera. Either tag the primary camera as main camera or set it on the Camera property on WeatherMakerScript.Instance. Multiple cameras in 3D are supported and the sky sphere, rain, fog, etc. will render around each camera. Add any additional cameras to the Cameras property of WeatherMakerScript.Instance.
- (Optional) Create your own custom script to manipulate Weather Maker. See WeatherMakerConfigurationScript.cs for an example.
- (Optional) Turn on the weather manager for automatic weather. See the "Weather Manager" section down below for details.
- For troubleshooting/performance, see that section at the very bottom of this document.
- For Integrations with 3rd party assets or to integrate into your own asset, see the Integrations section at the bottom of this file.

*** IMPORTANT ***
Weather Maker contains a built in GUI to get you started. This GUI lets you control most parameters of the asset. For your final release, you'll want to disable the ConfigurationCanvas object in the WeatherMakerPrefab object, and disable the WeatherMakerConfiguration script on the WeatherMakerPrefab object.
During play mode on a device with a keyboard, Ctrl+` will toggle the configuration panel visibility.

Tutorial videos:
- Weather Manager: https://www.youtube.com/watch?v=jCuAr6y0kxU
- Day/Night Cycle: https://www.youtube.com/watch?v=M6PTyr52a00
- Scene / Ambient Sounds: https://www.youtube.com/watch?v=LdnALY4eCU4
- Sky Sphere:
	Setup: https://www.youtube.com/watch?v=QE3VZHWkVec
	Procedural Sky: https://www.youtube.com/watch?v=sB7U-yz-i6k
	Clouds: https://www.youtube.com/watch?v=DWF8gjDIkeM
	Cloud Masking: https://www.youtube.com/watch?v=WAy-pGdfV3g
	Custom Skybox / Cloud Only Sky Sphere: https://www.youtube.com/watch?v=V3OpB1S2r4w
- Suns and Moons: https://www.youtube.com/watch?v=neVZMeljYIQ
- Multiple Cameras: https://www.youtube.com/watch?v=6y5U37p4RpE
- Fog:
	Fog Shadow Map Lighting: https://www.youtube.com/watch?v=PTIC1oQzxno
	Volumetric Lighting and Fog: https://www.youtube.com/watch?v=D9MUloqUQjU
	Older Fog Videos:
		Full Screen Fog: https://www.youtube.com/watch?v=1_w9C8hWTXw
		Fog Volumes: https://www.youtube.com/watch?v=jJ_tx0Vog0o
- 3D Demo: https://www.youtube.com/watch?v=25XEdmHFXQY
- 2D Demo: https://www.youtube.com/watch?v=oX0Sa2IC2D4
- Rain Inside: https://www.youtube.com/watch?v=zFT2KVoR3ro

WeatherMakerScript
--------------------------------------------------
WeatherMakerScript.cs is the script that manages everything. The simplest way to get acquainted with how this works is to check out WeatherMakerConfigurationScript.cs. This is the script that the configuration panel uses to control all aspects of Weather Maker.

Here is an example that creates rain, clouds and wind:

WeatherMakerScript.Instance.Precipitation = WeatherMakerPrecipitationType.Rain;
WeatherMakerScript.Instance.PrecipitationIntensity = 0.2;
WeatherMakerScript.Instance.Clouds = WeatherMakerCloudType.Heavy;
WeatherMakerScript.Instance.WindIntensity = 0.1f;

Weather Profiles
--------------------------------------------------
Weather Maker 2.8.0 added Weather Profiles, an easy and robust way to set all of the common weather properties like clouds, precipitation, fog and lightning. Profiles can be created during play mode and the changes will be persisted. See the WeatherProfile property on WeatherMakerScript.cs.

A number of pre-built profiles are included, check the Prefab/Profiles folder.

Weather Manager - Automatic Weather
--------------------------------------------------
Weather Maker 2.0 introduced the weather manager. This set of scripts allows automation of the weather using advanced script property manipulation. Automatic weather is disabled by default.

Weather manager is currently only available in 3D mode.

// To turn on the rainy weather manager, do the following:
WeatherMakerScript.Instance.ActivateWeatherManager(0); // activate the first weather manager, by default this is the rainy weather manager

// To switch to a different weather manager:
WeatherMakerScript.Instance.ActivateWeatherManager(1); // activate the second manager, by default this is the snowy weather manager

// Note you can setup and configure your own weather managers however you like. See the tutorial video link above under "Tutorial Videos".

Precipitation
--------------------------------------------------
Weather Maker comes with 4 precipitation prefabs: Rain, snow, hail and sleet. Each are very similar in their setup, but contain different material, textures and parameters.
Rain prefab contains a secondary "torrential rain" particle system, but this may not perform well on low end mobile devices, so you can turn it off for those devices.

All the precipitation use WeatherMakerFallingParticleScript. Each has the concept of four stages: none, light, medium and heavy.

You can use a custom precipitation prefab / script that you create by doing the following:
WeatherMakerScript.Instance.Precipitation = yourCustomPrecipitationScript;
WeatherMakerScript.Instance.Precipitation = WeatherMakerPrecipitationType.Custom;

Properties:
- Loop Source Light, Medium and Heavy: The audio source to play as the prefab changes to different stages. Each sound fades in and out nicely.
- Sound Intensity Threshold: Change at what point the intensity will go to a new stage. Intensity of 0 is always the none stage.
- Intensity: Change the overall intensity of the precipitation.
- Intensity Multiplier: Change the intensity even more - watch out for performance problems if you go above 1.
- Secondary Intensity Multiplier: Change the secondary intensity even more - watch out for performance problems if you go above 1.
- Mist Intensity Multiplier: Make the mist less or more intense - again watch out for performance problems if you go above 1.
- Base Emission Rate: Base number of particles to emit per second
- Base Emission Rate Secondary: Base number of secondary particles to emit per second
- Base Emission Rate Mist: Base number of mist particles to emit per second
- Particle System: The main particle system.
- Particle System Secondary: The secondary particle system, if available
- Mist Particle System: The mist particle system.
- Explosion Particle System: Only used in 2D. For 3D, the explosion particle system is a child object of the particle system.
- Secondary Threshold: Once the intensity gets higher than this, secondary particles (if available) start to appear.
- Mist Threshold: Once the intensity gets higher than this, mist starts to appear.

3D Properties:
- Height: The down-pour particles will start at this height from the camera.
- Forward Offset: Offset the particle system this many world units in front of the camera.
- SecondaryHeight: Secondary particles will start at this height from the camera.
- SecondaryForwardOffset: Offset secondary particles this many world units in front of the camera.
- Mist Height: How high the mist will start falling from.
- For collisions, the weather maker prefab itself has a collisions property.

2D Properties:
- Height Multiplier: Particles will start at y value of the screen height + (world height * value). A value of 0.15 would make the particles start 15% higher than the screen height.
- Width Multiplier: The particle field will be this much wider than the screen width.
- Collision Mask: What should the particles collide with? The weather maker prefab can control this value globally so don't change this.
- Collision Life Time: About how long particle should live when they collide.
- Explosion Emission Life Time Maximum: When particles collide and have a life time higher than this value, they may emit an explosion.
- Mist Collision Velocity Multiplier: The mist speed is multiplied by this value whenever it collides with something.
- Mist Collision Life Time Multiplier: The mist life time is multiplied by this value whenever it collides with something.

Virtual Reality
--------------------------------------------------
Weather Maker supports VR. For best performance, ensure that the stereo rendering mode is single pass (fast). Unity menu -> edit -> player -> other settings -> VR supported -> stereo rendering method.

Audio / Sound Volume Modifier
--------------------------------------------------
WeatherMakerScript.cs contains a VolumeModifierDictionary that determines the final volume for precipitation, wind and lightning. There are several sound dampening prefabs available that will reduce sounds when the player moves into them. The player must be tagged as 'Player' or you can set the required tag of the collider object on the sound dampening prefabs.

Weather Maker manages the volume transitions for you and handles overlapping sound dampening areas just fine.

Wind
--------------------------------------------------
This allows wind to randomly blow in your scene, which will affect the rain, snow, mist, etc. The WindScript property of the Weather Maker script controls the wind.

To enable or disable wind, set the WindIntensity property on WeatherMakerWindScript in the Wind prefab.

Wind affects fog velocity. This can be disabled by setting FogVelocityMultiplier to 0 on the wind script.

Wind direction is set either randomly, or using the transform of the Wind object. To disable random wind direction, set the WindMaximumChangeRotation property on the WeatherMakerWindScript to zeroes.

Please see the properties in WeatherMakerWindScript.cs for a full list of what is available. Each property has a tooltip description that explains the property.

Sun
--------------------------------------------------
The weather maker prefab contains a Sun object. It is recommended to only have one directional light, which should be this sun object. This object should stay enabled. It's also very important that the sun intensity not be changed by anything external the Weather Maker scripts, otherwise artifacts may occur.

If you need to change the sun intensity, the day night cycle script has a DirectionalLightIntensityMultipliers property which is a dictionary of a key and mutliplier that will affect sun and moon light intensity.

Although WeatherScript has a Suns array, only one sun is currently supported. Multiple suns may be supported in the future.

The sun must not be deactivated.

Moon
--------------------------------------------------
Earths moon is on by default in 3D mode. I don't have it always facing the right way, but it still looks pretty good. Accurate moon phases for the lat, lon, date and time are included.

Moon phase increases moon lighting intensity as well.

The moon is a directional light. Moon base intensity is set on the day night script. Moon lighting power and tint color, etc. is set on the sky sphere script.

Multiple moons are supported using the Moons property of WeatherMakerScript.Instance.

You can deactivate the moon objects if you don't want moons.

Clouds
--------------------------------------------------
In 3D, clouds are created using the cloud script. In 2D, clouds are created using the LegacyCloudScript2D property of the WeatherMakerScript class. See CloudToggleChanged in WeatherMakerConfigurationScript.cs for how to set clouds.

3D clouds used to be run on the sky sphere. This has changed and they are now a full screen command buffer, which will allow better performance and much nicer looking clouds in the future. Wherever you see sky sphere cloud properties being referenced, simply refer to the same properties in the cloud script.

The cloud noise texture can use up to 4 channels: r,g,b and a. A normal texture can also be supplied. The r,g,b,a of the noise texture denote the normal. A value of 0 means the normal is pointing down, otherwise the value denotes the amount of y axis rotation of the normal where 1 is full rotation.

The cloud script now uses a cloud profile as of Weather Maker 3.0.

Cloud Shadows:
3D clouds can cast shadows by using Unity projector. This object is disabled by default in the prefab, but you can enable it for shadows.
You can also turn off cloud shadows by setting CloudShadowThreshold to 1.0 on the cloud script.
Cloud shadows use a Unity projector. Set game object layer to TransparentFX or IgnoreRaycast if you don't want an object to have cloud shadows, or set the projector to the layers you wish to ignore.
Trees and grass on terrain don't receive projector. Feel free to bug Unity about this as it seems quite broken. I've tried custom grass shaders with no luck...
Cloud shadows take the sun light shadow strength into account, along with CloudShadowThreshold and CloudShadowPower on the sky sphere script.
Cloud shadows will increase in strength as cloud cover / density increases.
Cloud shadows also allows deactivating the lens flare for the sun if it is covered by clouds.

Lightning
--------------------------------------------------
- Located in WeatherMakerPrefab/ThunderAndLightningPrefab and WeatherMakerPrefab/ThunderAndLightningPrefab/LightningBoltPrefab.
- I've included the core of my Procedural Lightning asset (http://u3d.as/f1c) to power Weather Maker lightning.

ThunderAndLightningPrefab Setup
Ideally this script will work as is. Lightning will be created randomly in a hemisphere around the main camera. Lightning can also be forced visible, which means the script will try to make the bolt appear in the camera field of view.

Lightning light effects are less visible during the day and brighter at night. For best results, ensure that you create clouds before turning the lightning on.

This script has the concept of "Intense" vs. "Normal" lightning. Intense lightning will be created close to the player, be brighter and play a random intense sound very soon after the lightning strike, which will be loud. Normal lightning is further away, plays quieter sounds and those sounds take longer before playing.

WeatherMakerThunderAndLightning Properties:
- Lightning Bolt Script: The lightning bolt script used to generate the lightning. Leave as the default.
- Camera: The camera lightning should be created around. Default is the main camera.
- Lightning Interval Time Range: Random range in seconds between lightning.
- Lightning Intense Probability: Percent change that lightning will be intense (close and loud).
- Thunder Sounds Normal: List of sounds for normal lightning.
- Thunder Sounds Intense: List of sounds for intense lightning.
- Start Y Base: The base y value for lightning to start at.
- Start X Variance: Vary the x start position
- Start Y Variance: Vary the y start position
- Start Z Variance: Vary the z start position
- End X Variance: Vary the x end position
- End Y Variance: Vary the y end position. Lightning will ray-trace to the ground, so the end position may change.
- End Z Variance: Vary the z end position.
- Lightning forced visibility probability: The chance that the lightning bolt will attempt to be visible in the camera.
- Ground lightning chance: The chance that lightning will strike the ground.
- Cloud lightning chance: The chance that lightning will only be visible in the clouds.
- Sun: Used to lessen the lightning brightness during the day.
- Normal Distance: Force lightning away from the camera by this range of distance for normal lightning.
- Intense Distance: Force lightning away from the camera by this range of distance for intense lightning.
- *Note lightning will always be at least normal or intense distance minimum distance from the camera, regardless of Start / End variance settings.

Here are the properties you are most likely to want to change on LightningBoltPrefab (WeatherMakerLightningBoltPrefabScript):
- Duration Range: Range of how long, in seconds, that each lightning bolt will be displayed.
- Trunk width range: Range of possible width, in world units, of the trunk.
- Glow tint color: Change the outer color of the lightning.
- Generations: Change how many line segments the lightning has.
- Glow Intensity: How bright the glow is.
- Glow Width Multiplier: How wide the glow is.
- Lights: To turn off lights, set LightPercent to 0.

To get notified of lightning bolt start and end events, use WeatherMakerScript.Instance.LightningBoltScript.LightningStartedCallback and WeatherMakerScript.Instance.LightningBoltScript.LightningEndedCallback.

Sky Sphere
--------------------------------------------------
- Located in WeatherMakerPrefab/Sky/SkySphere.
- Not used in 2D mode, for 2D see the Sky Plane section.
- If you see any edges in the sky, increase the resolution property.

Sun / Moon:
- Celestial bodies are managed on WeatherScript, using the Suns and Moons properties. Only 1 sun is supported. Up to 8 moons are supported. Set the lighting properties on these scripts (Sun/Moons properties of WeatherMakerScript.cs), do not modify the Light object itself as it will get overwritten by the Weather Maker scripts. Orbit defaults to Earth based orbit, but can also be set to a custom orbit (see WeatherMakerScript.cs, OrbitTypeCustomScript property).
- Sun is rendererd with it's own mesh now, look for it in the prefab, just search for the Sun object.

SkyMode:
- Procedural: Sky is fully procedural and day and dawn dusk sky sphere textures are ignored. Night texture is used as the sun goes below the horizon.
- Procedural Textured: Default, blends a procedural sky with the textures you specify. Your day and dawn/dusk textures should contain transparent areas with translucent clouds, make sure the import settings for the textures allows transparency. Night texture should still be opaque. The night texture will be hidden by any areas of the dawn/dusk texture that are opaque.
- Textured: Sky is fully textured and not procedural. If you need to convert a cubemap to a sphere/panorama map, try this Google search: https://www.google.com/search?q=unity+cubemap+to+panorama&oq=unity+cubemap+to+pano&aqs=chrome.1.69i57j0.11805j0j4&sourceid=chrome&ie=UTF-8#q=unity+cubemap+to+equirectangular.
- Note: Setting the player color space to linear may give a more Earth-like sky color for procedural modes.

Texture Modes:
- If you are using SkyMode "Textured" or "Procedural Textured", there are texture options:
- Set texture type to "Advanced". Then disable all settings or set to defaults. Then set wrap mode to clamp, filter mode to bilinear, aniso level 1, set max size to appropriate value and format to automatic truecolor for best appearance.
- Sphere: Texture should be about 2:1 aspect ratio. Top half is upper part (sky), bottom half is the ground.
- Panorama: Entire texture is the upper half of the sky sphere only. Great for higher resolution sky when the player won't ever look below the horizon.
- Panorama Mirror Down: Same as Panorama, except that the texture mirrors down to the bottom half of the sky sphere.
- Dome: Texture maps such that the center of the texture is the top of the sky sphere, and the edges of the circle are the horizon. This produces the best looking sky but requires pre-processing of the texture before importing into Unity.
- Dome Mirror Down: Same as dome except that the texture mirros down to the bottom half of the sky sphere.
- Dome Double: Same as dome except the texture contains two domes. The left half is the bottom dome, the right half is the top dome.
- Fish Eye Mirrored: Maps the fish eye to the front of the sky sphere and mirrors it to the back. Not suitable for 360 degree viewing.
- Fish Eye 360: Maps the fish eye to the entire sky sphere. There is a slight distortion at one end of the sky sphere so this is not suitable for full 360 degree viewing.
- *NOTE* If you will be using the procedural sky option, you can null out the day and dawn/dusk texture properties to save some disk space.
All sky sphere modes come with example textures (WeatherMaker/Prefab/Textures/SkySphere).

Night Sky:
- Night sky intensity can be changed via script parameter on the sky sphere script.
- NightVisibilityThreshold controls which pixels are visible. For a city or high light pollution scene, you could raise this value so only bright stars are visible.
- The night sky stars twinkle by default. There are several parameters that control how the sky twinkles. See the night sky twinkle section in the inspector of the sky sphere script. You can turn off the twinkle by setting the randomness and twinkle variance to 0.

Sky Plane
--------------------------------------------------
In 2D, a Sky Plane is used for procedural sky. Most parameters are the same as the sky sphere.

The Sky Plane renders in the default sorting queue at int.MinValue.

This is a work in progress, so keep checking for updates and see DemoScene2D for an example.

Day Night Cycle
--------------------------------------------------
WeatherMakerDayNightCycle.cs contains full sun path functionality.

Sun - The directional light representing the sun, which is set to the Sun property of the Weather Maker script.
Speed - The speed of the cycle during the day. Negative values are allowed. A value of 1 matches real time.
NightSpeed - The speed of the cycle at night. Negative values are allowed, but should match the sign of the Speed parameter. A value of 1 matches real time.
UpdateInterval - How often (in seconds) the cycle updates. 0 is the default which is every frame. Increase this if you are seeing flickering shadows, etc.
TimeOfDay - The time of day in local time, in seconds.
Year - the current year, this is currently only used for sun positioning.
Month - the current month, this is currently only used for sun positioning.
Day - the current day of month, this is currently only used for sun positioning.
TimeZoneOffsetSeconds - Used to convert TimeOfDay to UTC time. You should look this up once and set it based on your longitude, year, month and day. Set to -1111 for auto-detect. During editor mode, -1111 will cause a web service query to get you the right timezone for you location and date.
Latitude - Latitude on the planet in degrees (0 - 90)
Longitude - Longitude on the planet in decimal degrees (-180 to 180)
AxisTilt - The tilt of the planet in degrees (earth is about 23.439 degrees)
RotateYDegrees - Rotate around the up vector, useful for something besides an East/West cycle
SunDotFadeThreshold - Begin fading in/out the sun when the dot product of it's direction and the down vector becomes less than or equal to this value.
SunDotDisableThreshold - Disable the sun when the dot product of it's direction and the down vector becomes less than or equal to this value. Useful to prevent artifacts when a directional light shines up through your scene.

Ambient Colors:
- Day, dusk and night ambient colors and intensities are confiurable in the day/night script.

You can change the sun intensity via the SunIntensityMultipliers dictionary. Use your own unique string and add a multiplier to change the sun intensity, which has a base value of 1. For example, the cloud scripts and sky sphere add their own intensity multipliers depending on what types of clouds are showing. Storm clouds dim the sun, etc.

Ambient Sounds / Weather Sounds
--------------------------------------------------
You can create any number of sounds to play during dawn, day, dusk and night or even particular hours in the day and/or in zones. Weather Maker uses scriptable objects to allow you to setup a sound profile.

Right click in the project window and select create -> Weather Maker -> Sound Group, and name your profile. Then do the same and create Sound objects for each type of sound you want. In the sound you can set the audio clip, whether it loops, interval, duration and fade time. See WeatherMakerSoundScript.cs for more details.

When you are ready to add your sound groups to Weather Maker, simply drag in WeatherMakerSoundZone prefab into your scene. If the sounds will be active in the whole scene, put the prefab inside your player object (so the trigger will always be entered), otherwise find a nice place in your scene for the particular sounds in the group you created.

The sounds are not meant to be 3D and will have the chance to play as long as the trigger zone has been entered.

To debug sounds, uncomment the first line in WeatherMakerSoundScript.cs which will log start / stop sounds to the console.

As always, please watch the sound tutorial in the tutorial list at the top of this file.

Fog
--------------------------------------------------
Fog is only supported in 3D currently.
Turn fog on and off by calling WeatherMakerScript.FogScript.EnableFog(fromIntensity, toIntensity, transitionSeconds);
Highlights:
- Highly performant and very configurable.
- Play around with the noise settings, density and fog types to get the look you want.
- You can set the fog height for ground fog.
- The noise scale, noise adder and noise multiplier will determine the variance and appearance of the fog. Please set these to appropriate values for the scale and feel of your game.
- *NOTE*: Fog noise height and warping has been removed in favor of true 3D fog noise. Set the fog adder property into the negatives to get a more varied fog appearance with holes.
- *NOTE*: For fog noise to work properly, it is important that the WeatherMakerLightManagerScript 'NoiseTexture3D' property is set. If you don't need fog noise, you can null out this property and save a few MB of disk space.
- FogNoiseSampleCount defines how many samples are taken for fog noise. You may need to adjust this for your platform, especially if you are targetting mobile devices.
- Fog contains shadow map capability. If fog shadow sample count is > 0, fog shadows will be used for sun only. Be careful about performance, and tune the settings. Watch the tutorial (see link at top) for best results.
- For full screen fog, a sun shaft sample count > 0 will add sun shafts. Tweak sample count and other parameters to your liking. 32 is a good value I have found.
- Full screen fog takes null fog zone prefab into account, clearing out fog using a box collider. Rotation is not supported. Up to 8 can be visible by default. Fog null zones must not overlap and the shader opts for performance, so null zones that are close together may have some slight fog artifacts. Null zones are also not applied to fog lights in the volumetric lighting for performance.
- Tweak the dithering level if you see banding in low density fog.
- Please watch the tutorial video (at the top of this readme file). It covers most parameters.
- Email support@digitalruby.com with questions.

If you are seeing rendering artifacts, try changing the FogRenderQueue of the WeatherMakerFullScreenFogScript.cs in the inspector. It's at the bottom in the full screen settings.

A volumetric fog sphere and fog cube prefab is available to try out, and are in the DemoScene under the world object, but are deactivated by default. The fog sphere and cube require a WeatherMakerPrefab be in your scene and setup properly.

Volumetric Lights:
Point and spot lights can be enabled in the fog by setting EnableFogLights to true. Lights must be added to the light manager (please see the light manager section). The easiest way is to add your lights to the light manager script 'AutoAddLights' list.

Please test that performance is adequate before shipping your game with this turned on. The demo scene is setup automatically to add the flash light, point light and a spot light. Volumetric lights work best in Linear color space.

Orthographic Size
--------------------------------------------------
For 2D Weather Maker, clouds and lightning might need a little tweaking, depending on your orthographic size. Please be sure to set the cloud anchor offset along with the lightning trunk width and light range to appropriate values for your orthographic size.

Light Manager
--------------------------------------------------
In 3D mode, Weather Maker contains a LightManager script. This script can keep track of the lights you want to use in the various Weather Maker shaders such as fog, clouds, etc. in WeatherMakerLightManagerScript.cs, SetLightsToShader sets up all the shader parameters for the lights in world space. These are set as global shader variables, so you can use them in your own shaders if you like.

The light manager automatically adds all sun, moon and lightning lights, you do not need to do this yourself.

The light manager can auto-find all lights in your scene, but this may be a performance problem, so if it is, you must manually add and remove lights that you want to be used for cloud and fog lighting, etc. or add them to the AutoAddLights list.

Shaders
--------------------------------------------------
Weather Maker contains a number of shaders. Some of the global shader variables may be useful to you if you have your own customer shaders. See WeatherMakerShader.cginc for a list of the globals and information about them.

Integrations
--------------------------------------------------
If you are integrating Weather Maker into your own asset, use a #if WEATHER_MAKER_PRESENT ... #endif around your code.

Weather Maker has integration with some other assets. Most integrations are available under the "Integrations" object underneath the prefab.
- UNET (Unity Networking)
	- Under extensions in the prefab is a UNet object. It is deactivated by default. Activate it to use Unity networking. This will sync weather profiles, weather manager and time of day from the server to all clients. Anything else you want synced will need to be done by you manually. Please feel free to suggest additional things to sync.
- AQUAS - water asset
	- You must define AQUAS_PRESENT under player settings -> other settings -> scripting define symbols.
	- If you are using fog, set the full screen fog render queue to after forward alpha. AQUAS water does not work if it renders before the transparent queue, so this is currently the only workaround.
	- You can set cloud specular and reflection reduction on the AQUAS integration script.
- Uber - standard shader ultra
	- You must define UBER_STANDARD_SHADER_ULTRA_PRESENT under player settings -> other settings -> scripting define symbols.
	- Add an UBER_GlobalParams script somewhere in your scene and turn off the "User Particle System" option.
	- Configure global params to get the effect you want.
	- Set the minimum water level and minimum wetness on the WeatherMakerExtensionUberScript on the Weather Maker prefab -> extensions object.
	- Weather Maker will set RainIntensity, WaterLevel, WetnessAmount and SnowLevel automatically.
- RTP - Relief Terrain Pack
	- You must define RELIEF_TERRAIN_PACK_PRESENT under player settings -> other settings -> scripting define symbols.
	- Weather maker will ...
- CTS (Complete Terrain Shader):
	- Weather Maker will set the rain and snow on the terrain when there is precipitation and set the season based on latitude/longitude/date/time.
	- You can control which of these features are on.
- Gaia
	- In the Gaia manager under the GX tab, Digital Ruby, LLC -> Weather Maker are a number of useful options including setting location, moon and/or time of day. This is only available in the Unity editor.

Performance Considerations / Troubleshooting
--------------------------------------------------
Here is a list of things to try if performance or behavior has problems:

- Ensure you are using the latest Unity version - 5.6.2+.
- Lower the emission value of the particle systems, especially snow, which is the most CPU intensive.
- Turn off mist by rasing the mist threshold to 1.
- Turn off per pixel lighting by setting WeatherMakerScript.PerPixelLighting to false.
- Set EnableFogLights to false on the fog scripts.
- Turn off collisions for particles (CollisionEnabled) property of the WeatherMakerPrefab object.
- Reduce lightning generations and turn off lights (WeatherMakerPrafab/ThunderAndLightningPrefab/LightningBoltPrefab).
- Turn off lightning glow (set glow intensity to 0).
- Turn off soft particles in quality settings.
- Double check that your terrain and collidable objects are setup as efficient as possible. For example, very large terrain can cause performance issues.
- Reduce the MaximumLightCount constant in WeatherMakerLightManagerScript.cs if GPU is pegged and you have many lights in your scene.
- Turn off CheckForEclipse on the sky sphere if you are seeing CPU spikes. These can be caused by a Unity bug.
- If rain, snow, etc. is flying up instead of down, double check all the velocity over time values on the particle systems. Everything except the explosions should have negative y values.
- Fog is not showing or is all white - Double check that the WeatherMakerPrefab is setup in your scene properly and the Sun object is enabled.
- Turn off sun shafts (set sample count to 0 on full screen fog script), or reduce sun shaft sample count and reduce sun shaft down sampling value.
- If you see shadow flickering, try adding some wind to vary the trees up a bit or increase the day CelestialObjectRotationUpdateInterval value on the day night script. For the fog, consider disabling shadows if it is enabled, or lowering the shadow map resolution in quality settings.
- If you have a dawn/dusk or night scene, I would strongly recommend enabling the post processing stack for Unity and turning on temporal anti-aliasing.

--------------------------------------------------
Known Issues:
- Sky sphere may have some slight distortion at the poles when using sphere or panorama mode. This is easily fixed by using dome or double dome mode, or by correcting the texture.
- Fish Eye 360 has some distortion at the side pole of the sphere. I have as of yet been unable to fix this. I welcome suggestions and feedback on how to fix this.
- Snow particle system with collision can be very intensive CPU wise. Be careful with the number of particles or turn off collisions.
- Sun can shine through the bottom as it dips below the horizon. Fix this by adding a large shadow casting cube as the base of your world. Find the LargeGroundAndSunHorizonBlocker object in the demo scene for an example.

I'm Jeff Johnson, I created Weather Maker just for you. Please email support@digitalruby.com with any feedback, bug reports or suggestions.

- Jeff

Credits:
http://soundbible.com/1718-Hailstorm.html
http://blenderartists.org/forum/archive/index.php/t-24038.html
https://www.binpress.com/tutorial/creating-an-octahedron-sphere/162
http://freesound.org/
http://www.orangefreesounds.com/meditation-music-for-relaxation-and-dreaming/