import io
import os
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="Studio Vision | Interior Design Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize GenAI Client
@st.cache_resource
def get_genai_client():
    # Supports Google AI Studio API key or Vertex AI credentials
    return genai.Client()

client = get_genai_client()

# --- SIDEBAR: Design Controls & Presets ---
st.sidebar.title("🛠️ Design Controls")

mode = st.sidebar.radio(
    "Generation Mode",
    ["Full Space Redesign (Image-to-Image)", "Targeted Zone Inpainting (Brush Mask)"]
)

lighting_preset = st.sidebar.selectbox(
    "Lighting Preset",
    [
        "Warm 2700K Architectural Cove & Downlights",
        "Overcast Natural Daylight (Soft Diffuse)",
        "Bright Golden Hour Sunlight with Cast Shadows",
        "Commercial 4000K Clean Workspace Illumination",
        "Moody Hospitality Evening (Low Ambient, High Accent)"
    ]
)

finish_presets = st.sidebar.multiselect(
    "Core Material Palette",
    [
        "Honed Microcement Floor",
        "Fluted White Oak Paneling",
        "Brushed Gunmetal / Brass Accents",
        "Calacatta Viola Marble",
        "Textured Bouclé & Linen Upholstery",
        "Acoustic Slatted Timber Ceiling",
        "Exposed Industrial Concrete"
    ]
)

aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["16:9", "4:3", "1:1", "3:4", "9:16"], index=0)
guidance_scale = st.sidebar.slider("Style Adherence (Guidance Scale)", 1.0, 15.0, 8.5, 0.5)

# --- MAIN WORKSPACE ---
st.title("🏛️ Interior Design Pitch Visualizer")
st.caption("Upload site photos or 3D clay sketches, mask zones to alter, and apply custom joinery & finishes.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Base Space & Annotations")
    uploaded_base = st.file_uploader("Upload Base Site Photo / Sketch", type=["jpg", "jpeg", "png"])
    
    uploaded_refs = st.file_uploader(
        "Upload Material / FF&E Swatches (Optional)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    base_image = None
    mask_image = None

    if uploaded_base:
        base_image = Image.open(uploaded_base).convert("RGB")
        # Resize for canvas responsiveness while keeping aspect ratio
        max_canvas_width = 650
        w_percent = max_canvas_width / float(base_image.size[0])
        h_size = int((float(base_image.size[1]) * float(w_percent)))
        canvas_img = base_image.resize((max_canvas_width, h_size), Image.Resampling.LANCZOS)

        if mode == "Targeted Zone Inpainting (Brush Mask)":
            st.info("🖌️ Paint over the areas you want to replace (e.g., floor, new joinery wall, ceiling lighting).")
            brush_size = st.slider("Brush Size", 5, 60, 25)
            
            # Interactive brush canvas overlay
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1.0)",
                stroke_width=brush_size,
                stroke_color="#FFFFFF",
                background_image=canvas_img,
                update_streamlit=True,
                height=h_size,
                width=max_canvas_width,
                drawing_mode="freedraw",
                key="canvas",
            )

            # Process mask data
            if canvas_result.image_data is not None:
                mask_array = canvas_result.image_data[:, :, 3]  # Extract alpha channel
                # Binarize mask (0 or 255)
                mask_binary = np.where(mask_array > 0, 255, 0).astype(np.uint8)
                raw_mask = Image.fromarray(mask_binary, mode="L")
                mask_image = raw_mask.resize(base_image.size, Image.Resampling.NEAREST)
        else:
            st.image(base_image, caption="Base Space Reference", use_container_width=True)

with col_right:
    st.subheader("2. Design Brief & Generation")
    
    design_notes = st.text_area(
        "Design Directives / Pitch Brief",
        placeholder="E.g., Replace the partition with full-height bespoke fluted oak storage joinery with recessed bronze handles. Add a curved modular bouclé sofa in the center and replace flooring with microcement.",
        height=140
    )

    generate_btn = st.button("✨ Generate Pitch Render", type="primary", use_container_width=True)

    if generate_btn:
        if not uploaded_base:
            st.warning("Please upload a base space photo or sketch first.")
        else:
            with st.spinner("Step 1/2: Gemini analyzing spatial perspective & material palette..."):
                # Prepare moodboard images
                ref_pil_images = [Image.open(f).convert("RGB") for f in uploaded_refs] if uploaded_refs else []

                # Multimodal architectural analysis prompt
                palette_str = ", ".join(finish_presets) if finish_presets else "Specified in directives"
                gemini_prompt = f"""
                You are a senior architectural visualizer. 
                Task: Analyze the perspective, eye-level camera height, natural lighting angles, and room geometry of the base image.
                Incorporate the provided material reference images, the target palette ({palette_str}), and the lighting preset ({lighting_preset}).
                Design Directives: {design_notes if design_notes else "Elevate the space with contemporary high-end finishes."}
                
                Produce a single, detailed, photorealistic architectural photography rendering prompt for an image generation model. 
                Focus on: exact materials, millwork detailing, shadow depths, specular reflections, and realistic scale. 
                Do not include preamble; output only the final descriptive prompt.
                """

                # Call Gemini for prompt synthesis
                gemini_inputs = [base_image, *ref_pil_images, gemini_prompt]
                gemini_resp = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=gemini_inputs
                )
                expanded_prompt = gemini_resp.text.strip()

            with st.expander("🔍 Expanded Architectural Prompt (Generated by Gemini)"):
                st.write(expanded_prompt)

            with st.spinner("Step 2/2: Generating photorealistic renders with Imagen..."):
                try:
                    # Final synthesis with Imagen 3
                    full_prompt = (
                        f"{expanded_prompt}, photorealistic architectural photography, 8k resolution, "
                        f"architectural digest interior, accurate bounce light, physically based rendering"
                    )

                    result = client.models.generate_images(
                        model="imagen-3.0-generate-002",
                        prompt=full_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=2,
                            aspect_ratio=aspect_ratio,
                            guidance_scale=guidance_scale,
                            person_generation="dont_allow"
                        )
                    )

                    st.success("Render Complete!")
                    
                    # Display results in grid
                    res_col1, res_col2 = st.columns(2)
                    cols = [res_col1, res_col2]
                    
                    for idx, generated_img in enumerate(result.generated_images):
                        img_bytes = generated_img.image.image_bytes
                        pil_res = Image.open(io.BytesIO(img_bytes))
                        with cols[idx]:
                            st.image(pil_res, caption=f"Option {idx + 1}", use_container_width=True)
                            st.download_button(
                                label=f"⬇️ Download Option {idx + 1}",
                                data=img_bytes,
                                file_name=f"pitch_render_v{idx + 1}.png",
                                mime="image/png",
                                key=f"dl_{idx}"
                            )

                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
