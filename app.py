import streamlit as st
import torch
import tiktoken
from pathlib import Path

from model import GPT, GPTConfig


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="My SLM",
    page_icon="🧠",
    layout="centered"
)


# -----------------------------
# MODEL PATH
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model_params.pt"


# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = GPT(GPTConfig()).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    return model, tokenizer, device


# -----------------------------
# UI
# -----------------------------

st.title("🧠 My Small Language Model")

st.write(
    "Enter a prompt and let your trained SLM generate text."
)


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    max_tokens = st.slider(
        "Response Length",
        min_value=20,
        max_value=120,
        value=60,
        step=10
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.2,
        max_value=1.4,
        value=0.8,
        step=0.1
    )

    top_k = st.slider(
        "Top-K",
        min_value=5,
        max_value=100,
        value=40,
        step=5
    )

    st.divider()

    if torch.cuda.is_available():
        st.success("GPU Available")
    else:
        st.info("Running on CPU")


# -----------------------------
# PROMPT
# -----------------------------

prompt = st.text_area(
    "💬 Enter your prompt",
    placeholder="Once upon a time...",
    height=150
)


# -----------------------------
# GENERATE BUTTON
# -----------------------------

if st.button(
    "✨ Generate Text",
    use_container_width=True
):

    if not prompt.strip():

        st.warning(
            "Please enter a prompt first."
        )

    elif not MODEL_PATH.exists():

        st.error(
            "best_model_params.pt not found."
        )

    else:

        try:

            with st.spinner(
                "Generating text..."
            ):

                model, tokenizer, device = load_model()

                # Convert prompt into tokens
                token_ids = tokenizer.encode(
                    prompt
                )

                input_ids = torch.tensor(
                    [token_ids],
                    dtype=torch.long,
                    device=device
                )

                # Generate
                with torch.inference_mode():

                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )

                # Convert tokens back to text
                generated_text = tokenizer.decode(
                    output_ids[0].tolist()
                )

            st.subheader("📝 Generated Text")

            st.write(generated_text)

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )