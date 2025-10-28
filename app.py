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
import tavus_integration
load_dotenv()

# Define output directory for global access
out_dir = Path("generated")
out_dir.mkdir(exist_ok=True)

# Improved retry function for handling rate limiting and network errors
def retry_with_backoff(func, max_retries=5, initial_delay=2):
    """
    Retry a function with exponential backoff when rate limit errors occur
    or when network/server issues arise
    
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
            
            # Check if we should retry
            should_retry = False
            retry_reason = ""
            
            # Rate limit errors
            if "throttled" in error_message or "rate limit" in error_message or "too many requests" in error_message:
                should_retry = True
                retry_reason = "Rate limit"
            
            # Network/server errors
            elif "timeout" in error_message or "connection" in error_message or "disconnected" in error_message or "reset" in error_message:
                should_retry = True
                retry_reason = "Network error"
            
            # Handle retry logic
            if should_retry:
                if retries == max_retries:
                    st.error(f"{retry_reason} error persisted after {max_retries} retries. Please try again later.")
                    raise
                
                # Calculate delay with exponential backoff (2^retries * initial_delay)
                delay = (2 ** retries) * initial_delay
                st.warning(f"{retry_reason} detected. Waiting {delay} seconds before retry {retries+1}/{max_retries}...")
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
        scenes = st.slider("Number of scenes/pages", 1, 5, 1)

    # Add the Generate Book button to the first tab
    generate_book = st.button("Generate book")
    
    # Only run story generation code if the button is clicked
    if generate_book:
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
                        
                        # Check if base image exists
                        if not os.path.exists(base_image_path):
                            st.error(f"Base image not found at {base_image_path}. Falling back to text-to-image generation.")
                            # Fall back to base image generation if the file is missing (not a rate limit issue)
                            # In this case, we're handling a missing file, not a rate limit, so fall back is okay
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
                        
                        # Pass the file object directly to Replicate using a context manager
                        # to ensure the file is properly closed after the API call
                        with open(base_image_path, "rb") as image_file:
                                # Load the image to verify it's valid
                                try:
                                    # Check if image is valid by opening it
                                    img = Image.open(image_file)
                                    img.verify()  # Verify it's a valid image
                                    image_file.seek(0)  # Reset file pointer after verification
                                    
                                    # Make the API call
                                    return replicate.run(
                                        "stability-ai/stable-diffusion-img2img:ddd4eb440853a42c055203289a3da0c8886b0b9492fe619b1c1dbd34be160ce7",
                                        input={
                                            **input_payload,
                                            "image": image_file
                                        }
                                    )
                                except Exception as img_error:
                                    error_message = str(img_error).lower()
                                    
                                    # Check if it's a rate limit/throttling error
                                    if "throttled" in error_message or "rate limit" in error_message or "too many requests" in error_message:
                                        # For throttling errors, propagate the error to let retry_with_backoff handle it
                                        st.warning(f"Rate limiting detected: {str(img_error)}. Will retry img2img after delay...")
                                        raise img_error
                                    else:
                                        # Only fall back to text-to-image for non-throttling errors
                                        st.warning(f"Issue with image file: {str(img_error)}. Falling back to text-to-image generation.")
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
                        st.error(f"Error in image generation for scene {scene_num}: {str(e)}")
                        raise
                
                # Execute with retry logic
                output = retry_with_backoff(img2img_for_scene)
                
                # Simply use the output as a URL string
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
    # Define progress_status for this tab to avoid errors when switching tabs
    progress_status = st.empty()
    
    st.subheader("Talk with the Character")
    st.write("Create a video conversation with your character to practice language skills!")
    
    # UI elements for conversation
    if 'conversation_created' not in st.session_state:
        st.session_state.conversation_created = False
        st.session_state.conversation_id = None
        st.session_state.conversation_url = None
        st.session_state.story = None
    
    # Allow conversation creation with or without a story
    if not st.session_state.conversation_created:
        # Add story status message
        if 'story' in st.session_state and st.session_state.story:
            st.success("Using story generated in the 'Create Book' tab")
        else:
            st.info("No story generated. A default story context will be used.")
            
        # Add character settings section
        st.subheader("Character Settings")
        
        # If no story, get all character details
        if 'story' not in st.session_state or not st.session_state.story:
            direct_char_col1, direct_char_col2 = st.columns(2)
            with direct_char_col1:
                direct_child_name = st.text_input("Child's name for conversation", "Asha", key="direct_child_name")
                direct_target_language = st.selectbox("Conversation language", ["Spanish", "French", "English"], key="direct_target_lang")
            with direct_char_col2:
                direct_style = st.selectbox("Character style", ["cartoon monkey", "friendly robot", "cute cat"], key="direct_style")
        else:
            # If story exists, show the existing settings
            st.info(f"Using character '{style} {child_name}' and language '{target_language}' from the story.")
        
        # Create conversation button - always visible
            if st.button("Create Conversation"):
                with st.spinner("Creating conversation..."):
                    # Use the appropriate character details based on whether we have a story
                    if 'story' in st.session_state and st.session_state.story:
                        # Create Tavus conversation using the generated story
                        result = tavus_integration.initialize_tavus_conversation_from_story(
                            story_scenes=st.session_state.story,
                            character_name=f"{style} {child_name}",
                            target_language=target_language,
                            child_name=child_name
                        )
                    else:
                        # Create Tavus conversation with default story
                        result = tavus_integration.initialize_tavus_conversation_from_story(
                            character_name=f"{direct_style}",
                            target_language=direct_target_language,
                            child_name=direct_child_name
                        )
                    
                        # Store default story in session state for future reference
                        if not 'story' in st.session_state or not st.session_state.story:
                            st.session_state.story = tavus_integration.get_default_story(
                                direct_style, 
                                direct_child_name,
                                direct_target_language
                    )
                    
                    if "error" in result:
                        st.error(f"Error creating conversation: {result['error']}")
                    else:
                        # Store conversation details - the API returns "conversation_id", not "id"
                        st.session_state.conversation_id = result.get("conversation_id")
                    
                    # Show the result for debugging
                    st.write("Conversation API response:")
                    st.json(result)
                    
                    if not st.session_state.conversation_id:
                        st.error("No conversation ID returned from API. Expected 'conversation_id' field.")
                    else:
                        # Use the URL directly from the API response if available
                        original_url = result.get("conversation_url") or tavus_integration.get_conversation_url(st.session_state.conversation_id)
                        
                        # Check if it's a valid Tavus URL
                        if original_url:
                            # Create alternative URLs if needed
                            conversation_id = st.session_state.conversation_id
                            alternative_urls = [
                                original_url,
                                f"https://tavus.daily.co/{conversation_id}",
                                f"https://tavus.io/conversation/{conversation_id}"
                            ]
                            
                            # Store all URLs for potential fallback
                            st.session_state.alternative_urls = alternative_urls
                            st.session_state.conversation_url = original_url
                        
                        st.session_state.conversation_created = True
                        
                        # Show conversation details
                        st.info(f"Conversation ID: {st.session_state.conversation_id}")
                        st.info(f"Conversation URL: {st.session_state.conversation_url}")
                        
                        # Update the HTML template with the conversation URL
                        try:
                            # Read the HTML template
                            with open("conversation_embed.html", "r") as f:
                                html_content = f.read()
                            
                            # Replace the placeholder with the actual URL
                            html_content = html_content.replace("CONVERSATION_URL_PLACEHOLDER", st.session_state.conversation_url)
                            
                            # Write the updated HTML file
                            conversation_file = out_dir / "conversation.html"
                            with open(conversation_file, "w") as f:
                                f.write(html_content)
                            
                            st.session_state.conversation_html_path = str(conversation_file)
                            
                            # Show success and force a rerun to display the full UI
                            st.success("Conversation created successfully! Preparing interface...")
                            
                            # Show spinner while preparing to rerun
                            with st.spinner("Loading conversation interface..."):
                                time.sleep(1)  # Brief pause for UX
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error preparing conversation HTML: {str(e)}")
            
    else:
        # Show conversation details with a more prominent display
        st.markdown("## Your Conversation is Ready!")
        st.success(f"Conversation ID: {st.session_state.conversation_id}")
        
        # Make sure we have a valid URL
        if st.session_state.conversation_url and (
            st.session_state.conversation_url.startswith("http://") or 
            st.session_state.conversation_url.startswith("https://")
        ):
            st.write("### Interactive Conversation")
            
            # Option 1: Embedded iframe directly in Streamlit
            st.write("You can interact with your AI language tutor below:")
            
            # Create iframe container with proper height and width
            st.markdown("""
            <style>
            .iframe-container {
                position: relative;
                width: 100%;
                min-height: 600px;
                overflow: hidden;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Embed the iframe directly in the app
            st.markdown(f"""
            <div class="iframe-container">
                <iframe 
                    src="{st.session_state.conversation_url}" 
                    width="100%" 
                    height="600px" 
                    allow="camera; microphone; fullscreen; display-capture; autoplay" 
                    referrerpolicy="no-referrer"
                    style="border: none; border-radius: 8px;"
                ></iframe>
            </div>
            """, unsafe_allow_html=True)
            
            # As a backup, also use Streamlit's component method
            # try:
            #     st.components.v1.iframe(
            #         src=st.session_state.conversation_url,
            #         height=600,
            #         scrolling=True,
            #         width=800  # Use a specific pixel value instead of "100%"
            #     )
            # except Exception as e:
            #     st.error(f"Alternative embedding method failed: {str(e)}")
            #     # If both embedding methods fail, still provide the direct link
            #     st.warning("If the embedded conversation isn't working, please try the direct link below:")
            #     st.markdown(f"**[Open Conversation in New Tab]({st.session_state.conversation_url})**")
            
            # Information about how to use
            st.info("""
            **How to use the conversation:**
            1. Allow camera and microphone permissions when prompted
            2. Speak in simple, clear sentences in your language
            3. The AI character will respond and help you practice the language
            4. You can refresh the page if you encounter any issues
            """)
            
        else:
            st.error(f"Invalid conversation URL: {st.session_state.conversation_url}")
            st.info("Please try creating a new conversation.")
        
        if st.button("Create New Conversation"):
            st.session_state.conversation_created = False
            st.session_state.conversation_id = None
            st.session_state.conversation_url = None
            st.rerun()

