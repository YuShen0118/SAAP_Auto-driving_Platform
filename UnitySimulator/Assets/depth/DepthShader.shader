Shader "Custom/DepthGrayscale" 
{
	SubShader {  
        Tags { "RenderType"="Opaque" }  
          
        Pass{  
            CGPROGRAM  
            #pragma vertex vert  
            #pragma fragment frag  
            #include "UnityCG.cginc"  
              
            sampler2D _CameraDepthTexture;  
			// 顶点着色阶段： 顶点着色器直接接受mesh中的顶点，每个顶点会调用一次。
            struct v2f {  
               float4 pos : SV_POSITION;  
               float4 scrPos:TEXCOORD1;  
            };  
              
            v2f vert (appdata_base v){  
               v2f o;  
               o.pos = UnityObjectToClipPos (v.vertex); // 把局部坐标的定点，转到建材坐标，这里集合了你常用的变换。
               o.scrPos=ComputeScreenPos(o.pos);  // 计算屏幕坐标，存成 scrpos，这样就有了3D点到屏幕2D点的关系。
               return o;  // 固定返回值，是为了发回给后续的着色器用的。
            }  

			// 片元着色器：片元着色器接收过来当点着色器传过来的一组值（注意，这里不仅仅是mesh中的顶点，是两点插值过后的N个点）
            half4 frag (v2f i) : COLOR{  
				//对应顶点着色器的值，取其深度值, 转换成[0,1]内的线性变化深度值 
			   float depth = tex2Dproj(_CameraDepthTexture,UNITY_PROJ_COORD(i.scrPos)).r;
               float depthValue =Linear01Depth (depth);
			   return half4(depthValue, depthValue, depthValue,1);
            }  
            ENDCG  
        }  
    }  
    FallBack "Diffuse"
}