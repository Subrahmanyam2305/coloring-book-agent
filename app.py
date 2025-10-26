# app.py
import os
import io
import base64
import time
import traceback
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import replicate
import streamlit as st
from dotenv import load_dotenv
import requests
import threading
# Import Tavus integration
import tavus_integration
load_dotenv()

# Retry function for handling rate limiting
def retry_with_backoff(func, max_retries=5, initial_delay=2):
    """
    Retry a function with exponential backoff when rate limit errors occur
    
    Args:
        func: The function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds before retrying
        
    Returns:
        The result of the function call
    """
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            return func()
        except Exception as e:
            error_message = str(e).lower()
            
            # Check if it's a rate limit error
            if "throttled" in error_message or "rate limit" in error_message or "too many requests" in error_message:
                if retries == max_retries:
                    st.error(f"Rate limit exceeded after {max_retries} retries. Please try again later.")
                    raise
                
                # Calculate delay with exponential backoff (2^retries * initial_delay)
                delay = (2 ** retries) * initial_delay
                st.warning(f"Rate limit hit. Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                retries += 1
            else:
                st.error(f"Error: {str(e)}")
                traceback.print_exc()
                raise
    
    return None  # This should not be reached if max_retries is positive

# init
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    st.error("Set REPLICATE_API_TOKEN in environment")
    st.stop()

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

st.title("Kid's Language Coloring Book — Prototype")

# Create tabs for different functionality
tab1, tab2 = st.tabs(["Create Book", "Interactive Conversation"])

# --- Book Creation Tab ---
with tab1:
    # --- Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        child_name = st.text_input("Child's name", "Asha")
        target_language = st.selectbox("Language", ["Spanish", "French", "English"])
        style = st.selectbox("Character style", ["cartoon monkey", "friendly robot", "cute cat"])
    with col2:
        seed_input = st.number_input("Random seed (0 = random)", min_value=0, max_value=2_147_483_647, value=0)
        scenes = st.slider("Number of scenes/pages", 1, 3, 1)

    # Add the Generate Book button to the first tab
    if st.button("Generate book"):
        # Set up progress tracking
        progress_status = st.empty()
        
        # Step 1: Generate story first
        progress_status.info("Step 1/4: Generating story with AI...")
        
        # Use Claude to generate a coherent story with multiple scenes
        story_prompt = f"""
        Create a short, engaging children's story with {scenes} scenes/pages about a character named {child_name} who is a {style}.
        The story should be appropriate for young children, positive, educational, and have a simple moral lesson.
        Structure the output as {scenes} numbered paragraphs, one for each scene/page.
        Each paragraph should be 2-3 sentences long - simple enough for a children's book.
        The story should have a beginning, middle, and resolution.
        """
        
        # Use Claude 3.7 Sonnet with streaming and retry logic
        st.text("Creating story...")
        story_placeholder = st.empty()
        
        # Define the streaming function to retry
        def stream_story():
            result_text = ""  # Local variable for building result
            
            # Stream with rate limit handling
            try:
                for event in replicate.stream(
                    "anthropic/claude-3.7-sonnet",
                    input={
                        "prompt": story_prompt,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "system_prompt": "You are an expert children's story writer. Create engaging, age-appropriate content.",
                        "extended_thinking": False
                    }
                ):
                    result_text += str(event)
                    story_placeholder.text(result_text)
                
                return result_text
            except Exception as e:
                st.error(f"Error in story generation: {str(e)}")
                raise
        
        # Execute with retry logic
        story_text = retry_with_backoff(stream_story)
        
        # Split into scenes/paragraphs and clean up
        story_scenes = []
        current_scene = ""
        for line in story_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new scene marker
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or line.startswith(('Scene', 'Page')):
                if current_scene:
                    story_scenes.append(current_scene)
                current_scene = line
            else:
                current_scene += " " + line
        
        # Add the last scene if it exists
        if current_scene:
            story_scenes.append(current_scene)
        
        # Limit to the number of scenes requested
        story_scenes = story_scenes[:scenes]
        
        # Step 2: Translate the story if needed
        progress_status.info("Step 2/4: Processing story content...")
        story = []
        if target_language != "English":
            progress_status.info(f"Step 2/4: Translating story to {target_language}...")
            for i, scene in enumerate(story_scenes):
                translation_prompt = f"Translate the following children's story text to {target_language}, keeping it simple and appropriate for young children: '{scene}'"
                
                # Use Claude 3.7 Sonnet with streaming for translation
                st.text(f"Translating scene {i+1}...")
                trans_placeholder = st.empty()
                
                # Define the streaming function to retry
                def stream_translation():
                    result_text = ""  # Local variable for building result
                    
                    try:
                        for event in replicate.stream(
                            "anthropic/claude-3.7-sonnet",
                            input={
                                "prompt": translation_prompt + " Keep the translation concise and appropriate for children.",
                                "max_tokens": 1024,
                                "temperature": 0.3,
                                "system_prompt": f"You are an expert {target_language} translator specializing in children's content.",
                                "extended_thinking": False
                            }
                        ):
                            result_text += str(event)
                            trans_placeholder.text(result_text)
                            
                        return result_text
                    except Exception as e:
                        st.error(f"Error in translation for scene {i+1}: {str(e)}")
                        raise
                
                # Execute with retry logic
                translated_text = retry_with_backoff(stream_translation)
                story.append(translated_text.strip())
        else:
            story = story_scenes
        
        # Story will be displayed alongside each image scene instead of all at once
        # Store story in session state for conversation tab
        st.session_state.story = story
        
        # Step 3: Create image prompts for each scene based on story content
        progress_status.info("Step 3/4: Creating image prompts based on story...")
        image_prompts = []
        
        # First, generate a description of the character for consistency
        character_description_prompt = f"""
        Create a detailed visual description of a cute, child-friendly {style} character named {child_name}.
        Describe specific features like colors, clothing, accessories, facial features, and any distinctive characteristics.
        This will be used to create consistent images across multiple scenes.
        Keep the description to 2-3 sentences focusing on visual elements only.
        """
        
        # Use Claude 3.7 Sonnet with streaming for character description
        st.text("Creating character description...")
        char_placeholder = st.empty()
        
        # Define the streaming function to retry
        def stream_character_desc():
            result_text = ""  # Local variable for building result
            
            try:
                for event in replicate.stream(
                    "anthropic/claude-3.7-sonnet",
                    input={
                        "prompt": character_description_prompt + " Keep your description very brief, ideally 2-3 sentences maximum.",
                        "max_tokens": 1024,
                        "temperature": 0.4,
                        "system_prompt": "You are an expert character designer. Create engaging, visually descriptive content.",
                        "extended_thinking": False
                    }
                ):
                    result_text += str(event)
                    char_placeholder.text(result_text)
                    
                return result_text
            except Exception as e:
                st.error(f"Error in character description generation: {str(e)}")
                raise
        
        # Execute with retry logic
        character_desc_text = retry_with_backoff(stream_character_desc)
        
        # Generate scene-specific image prompts
        for i, scene_text in enumerate(story_scenes):
            prompt_request = f"""
            Based on this story scene: "{scene_text}"
            
            And using this consistent character description: "{character_desc_text}"
            
            Create a detailed image prompt for a children's book illustration. The prompt should:
            1. Describe the scene setting, characters' positions, actions, and emotions
            2. Maintain visual consistency with the character description
            3. Be optimized for an image generation AI
            4. Include details that make it work well for a coloring book page
            5. Keep the style cute, colorful, kid-friendly with clean shapes and bold outlines
            
            Just return the prompt text without explanations or quotation marks.
            """
            
            # Use Claude 3.7 Sonnet with streaming for image prompt
            st.text(f"Creating image prompt for scene {i+1}...")
            prompt_placeholder = st.empty()
            
            # Define the streaming function to retry
            def stream_image_prompt():
                result_text = ""  # Local variable for building result
                
                try:
                    for event in replicate.stream(
                        "anthropic/claude-3.7-sonnet",
                        input={
                            "prompt": prompt_request,
                            "max_tokens": 1024,
                            "temperature": 0.5,
                            "system_prompt": "You are an expert image prompt engineer. Create detailed, visually descriptive prompts.",
                            "extended_thinking": False
                        }
                    ):
                        result_text += str(event)
                        prompt_placeholder.text(result_text)
                        
                    return result_text
                except Exception as e:
                    st.error(f"Error in image prompt generation for scene {i+1}: {str(e)}")
                    raise
            
            # Execute with retry logic
            image_prompt = retry_with_backoff(stream_image_prompt)
            image_prompt = image_prompt.strip()
            
            # Add consistent style elements
            image_prompt += " Style: clean shapes, bold outlines, flat colors, kid-friendly, 4k"
            image_prompts.append(image_prompt)
        
        # Step 4: Generate images for each scene
        progress_status.info("Step 4/4: Generating scene illustrations...")
        out_dir = Path("generated")
        out_dir.mkdir(exist_ok=True)
        
        scene_images = []
        scene_outlines = []
        base_image_url = None  # Will store the URL of the first generated image
        base_image_path = None  # Will store the local path to the first image
        
        for i, image_prompt in enumerate(image_prompts):
            st.subheader(f"Scene {i+1}")
            
            # Display this scene's story text
            st.subheader("Story:")
            st.write(story[i])
            
            scene_num = i + 1
            st.info(f"Generating image for scene {scene_num}...")
            
            # Use seed + scene_num for consistency but variation
            scene_seed = (42 if seed_input == 0 else seed_input) + i
            
            # For the first scene, generate a base character image
            # For subsequent scenes, use img2img with the base image for consistency
            if i == 0:
                # First scene - generate base image
                def generate_base_image():
                    try:
                        return replicate.run(
                            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                            input={
                                "width": 512,
                                "height": 512,
                                "prompt": image_prompt,
                                "refine": "expert_ensemble_refiner",
                                "scheduler": "DPMSolverMultistep",
                                "num_outputs": 1,
                                "guidance_scale": 7.5,
                                "negative_prompt": "dark, scary, blurry, low quality, text, writing, watermark",
                                "prompt_strength": 0.8,
                                "num_inference_steps": 25,
                                "seed": scene_seed
                            }
                        )
                    except Exception as e:
                        st.error(f"Error in base image generation: {str(e)}")
                        raise
                
                # Execute with retry logic
                output = retry_with_backoff(generate_base_image)
                
                # Simply use the output as a URL string
                img_url = output[0]
                response = requests.get(img_url)
                pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
                
                # Save base image to use as init image for subsequent scenes
                base_image_path = out_dir / "base_character.png"
                pil.save(base_image_path)
                
            else:
                # Subsequent scenes - use img2img with the base image for consistency
                def img2img_for_scene():
                    try:
                        # Use the stable-diffusion-img2img model with local file
                        with open(base_image_path, "rb") as img_file:
                            # Convert image to base64 for local handling
                            image_bytes = img_file.read()
                        
                        # Set up input for stable-diffusion-img2img model
                        input_payload = {
                            "width": 512,
                            "height": 512,
                            "prompt": image_prompt,
                            "scheduler": "DPMSolverMultistep",
                            "num_outputs": 1,
                            "guidance_scale": 7.5,
                            "prompt_strength": 0.8,  # Same as the example (controls how much to preserve)
                            "num_inference_steps": 25,
                            "seed": scene_seed,
                            "negative_prompt": "dark, scary, blurry, low quality, text, writing, watermark"
                        }
                        
                        # Pass the image bytes directly
                        # Replicate automatically handles file uploads for binary content
                        return replicate.run(
                            "stability-ai/stable-diffusion-img2img:ddd4eb440853a42c055203289a3da0c8886b0b9492fe619b1c1dbd34be160ce7",
                            input={
                                **input_payload,
                                "image": image_bytes
                            }
                        )
                    except Exception as e:
                        st.error(f"Error in image generation for scene {scene_num}: {str(e)}")
                        raise
                
                # Execute with retry logic
                output = retry_with_backoff(img2img_for_scene)
                
                # Handle output based on return type
                if hasattr(output[0], 'url'):
                    img_url = output[0].url()
                    response = requests.get(img_url)
                    pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
                elif hasattr(output[0], 'read'):
                    pil = Image.open(io.BytesIO(output[0].read())).convert("RGBA")
                else:
                    # Assume it's a URL string
                    img_url = output[0]
                    response = requests.get(img_url)
                    pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
            
            # Display the image
            st.image(pil, caption=f"Scene {scene_num} Illustration", use_container_width=True)
            
            # Convert to coloring page outline
            # convert RGBA -> RGB white bg
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil, mask=pil.split()[3] if pil.mode == "RGBA" else None)
            rgb = bg.convert("RGB")
            
            # use OpenCV for edge detection
            cv_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            # median blur to remove small details
            blurred = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, blockSize=9, C=2)
            # invert so outlines are black on white if needed (we want black outlines on white)
            outlines = cv2.bitwise_not(edges)
            
            pil_out = Image.fromarray(cv2.cvtColor(outlines, cv2.COLOR_BGR2RGB)).convert("L")
            st.image(pil_out, caption=f"Scene {scene_num} Coloring Page", use_container_width=True)
            
            # Save images for this scene
            rgb.save(out_dir / f"scene_{scene_num}_color.png")
            pil_out.save(out_dir / f"scene_{scene_num}_outline.png")
            
            # Add to collections
            scene_images.append(rgb)
            scene_outlines.append(pil_out)
            
            # Add separator between scenes
            st.markdown("---")
        
        # Save story text
        with open(out_dir / "story.txt", "w") as f:
            for i, scene in enumerate(story):
                f.write(f"Scene {i+1}:\n")
                f.write(scene + "\n\n")
                
        st.success("Done — check generated/ for outputs.")

# --- Interactive Conversation Tab ---
with tab2:
    st.subheader("Talk with the Character")
    st.write("Create a video conversation with your character to practice language skills!")
    
    # UI elements for conversation
    if 'conversation_created' not in st.session_state:
        st.session_state.conversation_created = False
        st.session_state.conversation_id = None
        st.session_state.conversation_url = None
        st.session_state.story = None
    
    # Show conversation controls when a story exists
    if 'story' in st.session_state and st.session_state.story:
        if not st.session_state.conversation_created:
            if st.button("Create Conversation"):
                with st.spinner("Creating conversation..."):
                    # Create Tavus conversation using the story
                    result = tavus_integration.initialize_tavus_conversation_from_story(
                        story_scenes=st.session_state.story,
                        character_name=f"{style} {child_name}",
                        target_language=target_language,
                        child_name=child_name
                    )
                    
                    if "error" in result:
                        st.error(f"Error creating conversation: {result['error']}")
                    else:
                        # Store conversation details
                        st.session_state.conversation_id = result.get("id")
                        st.session_state.conversation_url = tavus_integration.get_conversation_url(st.session_state.conversation_id)
                        st.session_state.conversation_created = True
                        
                        # Launch conversation in background thread
                        threading.Thread(
                            target=tavus_integration.join_conversation_room, 
                            args=(st.session_state.conversation_id, st.session_state.conversation_url),
                            daemon=True
                        ).start()
                        
                        st.success("Conversation created!")
        else:
            # Show conversation URL and embed iframe if available
            st.success(f"Conversation ready! ID: {st.session_state.conversation_id}")
            st.markdown(f"**[Open Conversation in New Tab]({st.session_state.conversation_url})**")
            
            # Embed the conversation if possible
            st.write("Conversation Preview:")
            st.components.v1.iframe(
                src=st.session_state.conversation_url,
                height=600,
                scrolling=True
            )
            
            if st.button("Create New Conversation"):
                st.session_state.conversation_created = False
                st.session_state.conversation_id = None
                st.session_state.conversation_url = None
                st.experimental_rerun()
    else:
        st.info("First generate a story in the 'Create Book' tab, then come back to create a conversation.")
    # Set up progress tracking
    progress_status = st.empty()
    
    # Step 1: Generate story first
    progress_status.info("Step 1/4: Generating story with AI...")
    
    # Use Claude to generate a coherent story with multiple scenes
    story_prompt = f"""
    Create a short, engaging children's story with {scenes} scenes/pages about a character named {child_name} who is a {style}.
    The story should be appropriate for young children, positive, educational, and have a simple moral lesson.
    Structure the output as {scenes} numbered paragraphs, one for each scene/page.
    Each paragraph should be 2-3 sentences long - simple enough for a children's book.
    The story should have a beginning, middle, and resolution.
    """
    
    # Use Claude 3.7 Sonnet with streaming and retry logic
    st.text("Creating story...")
    story_placeholder = st.empty()
    
    # Define the streaming function to retry
    def stream_story():
        result_text = ""  # Local variable for building result
        
        # Stream with rate limit handling
        try:
            for event in replicate.stream(
                "anthropic/claude-3.7-sonnet",
                input={
                    "prompt": story_prompt,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "system_prompt": "You are an expert children's story writer. Create engaging, age-appropriate content.",
                    "extended_thinking": False
                }
            ):
                result_text += str(event)
                story_placeholder.text(result_text)
            
            return result_text
        except Exception as e:
            st.error(f"Error in story generation: {str(e)}")
            raise
    
    # Execute with retry logic
    story_text = retry_with_backoff(stream_story)
    
    # Split into scenes/paragraphs and clean up
    story_scenes = []
    current_scene = ""
    for line in story_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new scene marker
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or line.startswith(('Scene', 'Page')):
            if current_scene:
                story_scenes.append(current_scene)
            current_scene = line
        else:
            current_scene += " " + line
    
    # Add the last scene if it exists
    if current_scene:
        story_scenes.append(current_scene)
    
    # Limit to the number of scenes requested
    story_scenes = story_scenes[:scenes]
    
    # Step 2: Translate the story if needed
    progress_status.info("Step 2/4: Processing story content...")
    story = []
    if target_language != "English":
        progress_status.info(f"Step 2/4: Translating story to {target_language}...")
        for i, scene in enumerate(story_scenes):
            translation_prompt = f"Translate the following children's story text to {target_language}, keeping it simple and appropriate for young children: '{scene}'"
            
            # Use Claude 3.7 Sonnet with streaming for translation
            st.text(f"Translating scene {i+1}...")
            trans_placeholder = st.empty()
            
            # Define the streaming function to retry
            def stream_translation():
                result_text = ""  # Local variable for building result
                
                try:
                    for event in replicate.stream(
                        "anthropic/claude-3.7-sonnet",
                        input={
                            "prompt": translation_prompt + " Keep the translation concise and appropriate for children.",
                            "max_tokens": 1024,
                            "temperature": 0.3,
                            "system_prompt": f"You are an expert {target_language} translator specializing in children's content.",
                            "extended_thinking": False
                        }
                    ):
                        result_text += str(event)
                        trans_placeholder.text(result_text)
                        
                    return result_text
                except Exception as e:
                    st.error(f"Error in translation for scene {i+1}: {str(e)}")
                    raise
            
            # Execute with retry logic
            translated_text = retry_with_backoff(stream_translation)
            story.append(translated_text.strip())
    else:
        story = story_scenes
    
    # Story will be displayed alongside each image scene instead of all at once
    # Store story in session state for conversation tab
    st.session_state.story = story
    
    # Step 3: Create image prompts for each scene based on story content
    progress_status.info("Step 3/4: Creating image prompts based on story...")
    image_prompts = []
    
    # First, generate a description of the character for consistency
    character_description_prompt = f"""
    Create a detailed visual description of a cute, child-friendly {style} character named {child_name}.
    Describe specific features like colors, clothing, accessories, facial features, and any distinctive characteristics.
    This will be used to create consistent images across multiple scenes.
    Keep the description to 2-3 sentences focusing on visual elements only.
    """
    
    # Use Claude 3.7 Sonnet with streaming for character description
    st.text("Creating character description...")
    char_placeholder = st.empty()
    
    # Define the streaming function to retry
    def stream_character_desc():
        result_text = ""  # Local variable for building result
        
        try:
            for event in replicate.stream(
                "anthropic/claude-3.7-sonnet",
                input={
                    "prompt": character_description_prompt + " Keep your description very brief, ideally 2-3 sentences maximum.",
                    "max_tokens": 1024,
                    "temperature": 0.4,
                    "system_prompt": "You are an expert character designer. Create engaging, visually descriptive content.",
                    "extended_thinking": False
                }
            ):
                result_text += str(event)
                char_placeholder.text(result_text)
                
            return result_text
        except Exception as e:
            st.error(f"Error in character description generation: {str(e)}")
            raise
    
    # Execute with retry logic
    character_desc_text = retry_with_backoff(stream_character_desc)
    
    # Generate scene-specific image prompts
    for i, scene_text in enumerate(story_scenes):
        prompt_request = f"""
        Based on this story scene: "{scene_text}"
        
        And using this consistent character description: "{character_desc_text}"
        
        Create a detailed image prompt for a children's book illustration. The prompt should:
        1. Describe the scene setting, characters' positions, actions, and emotions
        2. Maintain visual consistency with the character description
        3. Be optimized for an image generation AI
        4. Include details that make it work well for a coloring book page
        5. Keep the style cute, colorful, kid-friendly with clean shapes and bold outlines
        
        Just return the prompt text without explanations or quotation marks.
        """
        
        # Use Claude 3.7 Sonnet with streaming for image prompt
        st.text(f"Creating image prompt for scene {i+1}...")
        prompt_placeholder = st.empty()
        
        # Define the streaming function to retry
        def stream_image_prompt():
            result_text = ""  # Local variable for building result
            
            try:
                for event in replicate.stream(
                    "anthropic/claude-3.7-sonnet",
                    input={
                        "prompt": prompt_request,
                        "max_tokens": 1024,
                        "temperature": 0.5,
                        "system_prompt": "You are an expert image prompt engineer. Create detailed, visually descriptive prompts.",
                        "extended_thinking": False
                    }
                ):
                    result_text += str(event)
                    prompt_placeholder.text(result_text)
                    
                return result_text
            except Exception as e:
                st.error(f"Error in image prompt generation for scene {i+1}: {str(e)}")
                raise
        
        # Execute with retry logic
        image_prompt = retry_with_backoff(stream_image_prompt)
        image_prompt = image_prompt.strip()
        
        # Add consistent style elements
        image_prompt += " Style: clean shapes, bold outlines, flat colors, kid-friendly, 4k"
        image_prompts.append(image_prompt)
    
    # Step 4: Generate images for each scene
    progress_status.info("Step 4/4: Generating scene illustrations...")
    out_dir = Path("generated")
    out_dir.mkdir(exist_ok=True)
    
    scene_images = []
    scene_outlines = []
    base_image_url = None  # Will store the URL of the first generated image
    base_image_path = None  # Will store the local path to the first image
    
    for i, image_prompt in enumerate(image_prompts):
        st.subheader(f"Scene {i+1}")
        
        # Display this scene's story text
        st.subheader("Story:")
        st.write(story[i])
        
        scene_num = i + 1
        st.info(f"Generating image for scene {scene_num}...")
        
        # Use seed + scene_num for consistency but variation
        scene_seed = (42 if seed_input == 0 else seed_input) + i
        
        # For the first scene, generate a base character image
        # For subsequent scenes, use img2img with the base image for consistency
        if i == 0:
            # First scene - generate base image
            def generate_base_image():
                try:
                    return replicate.run(
                        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                        input={
                            "width": 512,
                            "height": 512,
                            "prompt": image_prompt,
                            "refine": "expert_ensemble_refiner",
                            "scheduler": "DPMSolverMultistep",
                            "num_outputs": 1,
                            "guidance_scale": 7.5,
                            "negative_prompt": "dark, scary, blurry, low quality, text, writing, watermark",
                            "prompt_strength": 0.8,
                            "num_inference_steps": 25,
                            "seed": scene_seed
                        }
                    )
                except Exception as e:
                    st.error(f"Error in base image generation: {str(e)}")
                    raise
            
            # Execute with retry logic
            output = retry_with_backoff(generate_base_image)
            
            # Handle output - can be URL string or object with read/url methods
            if hasattr(output[0], 'url'):
                img_url = output[0].url()
                response = requests.get(img_url)
                pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
            elif hasattr(output[0], 'read'):
                pil = Image.open(io.BytesIO(output[0].read())).convert("RGBA")
            else:
                # Assume it's a URL string
                img_url = output[0]
                response = requests.get(img_url)
                pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
            
            # Save base image to use as init image for subsequent scenes
            base_image_path = out_dir / "base_character.png"
            pil.save(base_image_path)
            
        else:
            # Subsequent scenes - use img2img with the base image for consistency
            def img2img_for_scene():
                try:
                    # Use the stable-diffusion-img2img model with local file
                    with open(base_image_path, "rb") as img_file:
                        # Convert image to base64 for local handling
                        image_bytes = img_file.read()
                    
                    # Set up input for stable-diffusion-img2img model
                    input_payload = {
                        "width": 512,
                        "height": 512,
                        "prompt": image_prompt,
                        "scheduler": "DPMSolverMultistep",
                        "num_outputs": 1,
                        "guidance_scale": 7.5,
                        "prompt_strength": 0.8,  # Same as the example (controls how much to preserve)
                        "num_inference_steps": 25,
                        "seed": scene_seed,
                        "negative_prompt": "dark, scary, blurry, low quality, text, writing, watermark"
                    }
                    
                    # Pass the image bytes directly
                    # Replicate automatically handles file uploads for binary content
                    return replicate.run(
                        "stability-ai/stable-diffusion-img2img:ddd4eb440853a42c055203289a3da0c8886b0b9492fe619b1c1dbd34be160ce7",
                        input={
                            **input_payload,
                            "image": image_bytes
                        }
                    )
                except Exception as e:
                    st.error(f"Error in image generation for scene {scene_num}: {str(e)}")
                    raise
            
            # Execute with retry logic
            output = retry_with_backoff(img2img_for_scene)
            
            # Handle output based on return type
            if hasattr(output[0], 'url'):
                img_url = output[0].url()
                response = requests.get(img_url)
                pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
            elif hasattr(output[0], 'read'):
                pil = Image.open(io.BytesIO(output[0].read())).convert("RGBA")
            else:
                # Assume it's a URL string
                img_url = output[0]
                response = requests.get(img_url)
                pil = Image.open(io.BytesIO(response.content)).convert("RGBA")
        
        # Display the image
        st.image(pil, caption=f"Scene {scene_num} Illustration", use_container_width=True)
        
        # Convert to coloring page outline
    # convert RGBA -> RGB white bg
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil, mask=pil.split()[3] if pil.mode == "RGBA" else None)
    rgb = bg.convert("RGB")

    # use OpenCV for edge detection
    cv_img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # median blur to remove small details
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=9, C=2)
    # invert so outlines are black on white if needed (we want black outlines on white)
    outlines = cv2.bitwise_not(edges)

    pil_out = Image.fromarray(cv2.cvtColor(outlines, cv2.COLOR_BGR2RGB)).convert("L")
    st.image(pil_out, caption=f"Scene {scene_num} Coloring Page", use_container_width=True)
    
    # Save images for this scene
    rgb.save(out_dir / f"scene_{scene_num}_color.png")
    pil_out.save(out_dir / f"scene_{scene_num}_outline.png")
    
    # Add to collections
    scene_images.append(rgb)
    scene_outlines.append(pil_out)
    
    # Add separator between scenes
    st.markdown("---")
    
    # Save story text
    with open(out_dir / "story.txt", "w") as f:
        for i, scene in enumerate(story):
            f.write(f"Scene {i+1}:\n")
            f.write(scene + "\n\n")

    st.success("Done — check generated/ for outputs.")
