import streamlit as st
import torch
from pathlib import Path
from packages.steering_runner.runner import SteeringRunner
from packages.lora_trainer.runner import LoRARunner
from packages.steering_extractor.model_loader import ModelLoader
from packages.steering_extractor.config import VECTORS_DIR

st.set_page_config(page_title="Stoic LLM Steering", page_icon="🏛️")

st.title("🏛️ Stoic Philosophy LLM Steering")
st.markdown("*Steering language models to write like ancient Stoic philosophers*")

# Initialize session state for tab tracking and prompts
if "current_tab" not in st.session_state:
    st.session_state.current_tab = None
if "caa_prompt" not in st.session_state:
    st.session_state.caa_prompt = "When facing difficulty, one should"
if "lora_prompt" not in st.session_state:
    st.session_state.lora_prompt = "When facing difficulty, one should"

# Create tabs
tab1, tab2 = st.tabs(["🎯 CAA Steering", "🔧 LoRA Fine-tuning"])

# ==================== CAA TAB ====================
with tab1:
    # Clear cache if switching from LoRA to CAA
    if st.session_state.current_tab == "lora":
        st.cache_resource.clear()
    st.session_state.current_tab = "caa"

    st.header("Contrastive Activation Addition")
    st.markdown(
        "Steer the model using activation differences between Stoic and neutral text"
    )

    # Warning about CAA limitations
    st.warning(
        "⚠️ **Note:** CAA shows experimental results. Declarative prompts work best. For more stable output, try the LoRA tab."
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("⚙️ Settings")

        caa_author_files = {
            "Marcus Aurelius": "marcus_aurelius_steering.pt",
            "Seneca": "seneca_steering.pt",
            "Epictetus": "epictetus_steering.pt",
            "Combined (Average)": "combined",
        }

        caa_author = st.selectbox(
            "Choose Philosopher:", list(caa_author_files.keys()), key="caa_author"
        )

        caa_coefficient = st.slider(
            "Steering Strength:",
            min_value=0.0,
            max_value=0.3,
            value=0.11,
            step=0.01,
            key="caa_coef",
        )

    with col2:
        st.markdown("**💡 Try these example prompts:**")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("💪 Facing Difficulty", use_container_width=True):
                st.session_state.caa_prompt = "When facing difficulty, one should"
        with col_b:
            if st.button("⚖️ Nature of Virtue", use_container_width=True):
                st.session_state.caa_prompt = "The nature of virtue is"
        with col_c:
            if st.button("🌱 Living Well", use_container_width=True):
                st.session_state.caa_prompt = "To live well means"

        caa_prompt = st.text_area(
            "📝 Or enter your own prompt:",
            value=st.session_state.caa_prompt,
            height=100,
            key="caa_prompt_input",
        )

        st.info(
            "💡 **Tip:** Declarative prompts work best. Avoid questions like 'Should I...' or 'What is...'"
        )

        if st.button("🎯 Generate (CAA)", type="primary"):
            with st.spinner("Generating with CAA steering..."):
                # Cache the model to match standalone behavior
                @st.cache_resource
                def get_caa_model():
                    loader = ModelLoader()
                    return loader.load()

                model, tokenizer = get_caa_model()

                # Load steering vector
                if caa_author == "Combined (Average)":
                    vectors = []
                    for file in [
                        "marcus_aurelius_steering.pt",
                        "seneca_steering.pt",
                        "epictetus_steering.pt",
                    ]:
                        vec = torch.load(
                            VECTORS_DIR / file, map_location="cpu", weights_only=True
                        )
                        vectors.append(vec)
                    steering_vector = sum(vectors) / len(vectors)
                    temp_path = VECTORS_DIR / "temp_combined.pt"
                    torch.save(steering_vector, temp_path)
                    vector_path = temp_path
                else:
                    vector_path = VECTORS_DIR / caa_author_files[caa_author]

                runner = SteeringRunner(
                    vector_path,
                    model,
                    tokenizer,
                    coefficient=caa_coefficient,
                    prompts=[caa_prompt],
                    max_tokens=150,
                )

                results = runner.run_model_with_hook(return_output=True)
                generated = results[0] if results else "Error generating text"

                st.markdown("### 📜 Generated Text")
                st.info(generated)

# ==================== LoRA TAB ====================
with tab2:
    # Clear cache if switching from CAA to LoRA
    if st.session_state.current_tab == "caa":
        st.cache_resource.clear()
    st.session_state.current_tab = "lora"

    st.header("LoRA Fine-tuned Models")
    st.markdown("Models fine-tuned directly on Stoic philosophical texts")

    st.success(
        "✅ **Recommended:** LoRA provides more stable and consistent Stoic-style generation"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("⚙️ Settings")

        lora_author_map = {
            "Marcus Aurelius": "marcus_aurelius",
            "Seneca": "seneca",
            "Epictetus": "epictetus",
        }

        lora_author = st.selectbox(
            "Choose Philosopher:", list(lora_author_map.keys()), key="lora_author"
        )

        lora_temp = st.slider(
            "Temperature:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="lora_temp",
        )

    with col2:
        st.markdown("**💡 Try these example prompts:**")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button(
                "💪 Facing Difficulty", key="lora_diff", use_container_width=True
            ):
                st.session_state.lora_prompt = "When facing difficulty, one should"
        with col_b:
            if st.button(
                "⚖️ Nature of Virtue", key="lora_virtue", use_container_width=True
            ):
                st.session_state.lora_prompt = "The nature of virtue is"
        with col_c:
            if st.button("🌱 Living Well", key="lora_live", use_container_width=True):
                st.session_state.lora_prompt = "To live well means"

        lora_prompt = st.text_area(
            "📝 Or enter your own prompt:",
            value=st.session_state.lora_prompt,
            height=100,
            key="lora_prompt_input",
        )

        if st.button("🔧 Generate (LoRA)", type="primary"):
            with st.spinner("Generating with LoRA model..."):

                @st.cache_resource
                def get_lora_runner():
                    return LoRARunner()

                lora_runner = get_lora_runner()

                generated = lora_runner.generate(
                    lora_author_map[lora_author],
                    lora_prompt,
                    temperature=lora_temp,
                    max_tokens=150,
                )

                st.markdown("### 📜 Generated Text")
                st.success(generated)

# ==================== INFO SECTION ====================
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
    This demo compares two approaches to steering language models toward Stoic philosophy:
    
    ### 🎯 Contrastive Activation Addition (CAA)
    - Extracts activation differences between Stoic and neutral text
    - Applies steering vector during generation at layer 12
    - Zero additional training required
    - Adjustable steering strength
    - **Limitations:** Results can be inconsistent on small models
    
    ### 🔧 LoRA Fine-tuning (Recommended)
    - Fine-tunes model directly on Stoic texts
    - Uses Parameter-Efficient Fine-Tuning (PEFT)
    - Only ~850k parameters trained (0.07% of model)
    - More stable and consistent generation
    - Better preservation of philosophical style
    
    ### 📊 Technical Details
    **Base Model:** Llama-3.2-1B  
    **Data Source:** Project Gutenberg classical texts  
    **Neutral Pairs:** Generated using Claude API  
    **Training Data:** 30 contrastive pairs per philosopher  
    
    ### 🏗️ Data Pipeline
    1. Extract classical Stoic texts from Project Gutenberg
    2. Filter religious content and bibliographic data
    3. Generate neutral paraphrases using Claude API
    4. Train steering vectors (CAA) or LoRA adapters
    5. Apply to base model during generation
    
    ### 🎓 Key Learnings
    - CAA is effective but sensitive to base model quality
    - Small models (1B params) struggle with style preservation
    - LoRA provides more reliable results for style transfer
    - Data quality is crucial for both approaches
    
    **GitHub:** [Add your repo link here]
    """
    )
