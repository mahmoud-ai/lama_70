from torch import cuda, bfloat16
import transformers
from translator import translate

model_id = 'meta-llama/Llama-2-70b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth, 
    local_files_only=True
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)


def create_dialogue(crisis, sector, is_injuries=0):
    crisis = translate(crisis, tgt_lang="english")
    sector = translate(sector, tgt_lang="english")

    sector_messages = {
        "health": f"It is imperative to consistently make decisions that safeguard the health of citizens, mitigate the risk of potential natural disasters, and offer recommendations aimed at preventing such occurrences.",  # Health sector guidance
        "national security": f"Your decisions should be geared towards ensuring Egyptian national security and the safety of the people, all while respecting international and diplomatic boundaries. ",  # National security sector guidance
        "economic": f"You are required to make decisions that serve the best interests of the nation's economy, considering prevailing economic conditions, the stock market, and the guidance of the Central Bank of Egypt.",  # Economic sector guidance
        "education": f" You are tasked with making decisions that prioritize the best interests of the students, fostering their learning experiences, and offering recommendations that contribute to the advancement of education. ",  # Education sector guidance
        "foreign policy": f" The decision must align accurately with the policies and laws of the Arab Republic of Egypt, ensuring it is in the best interest of the country, all the while upholding strong international diplomatic relations. ",  # Foreign policy sector guidance
        "media": f" Compliance with the regulations set forth by the unions associated with this sector is imperative, encompassing both audio-visual and written domains. "  # Media sector guidance
    }

    prompt_base = f"As an Egyptian political man Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis} , "
    if is_injuries:
        prompt_base += " Taking into account the presence of dead and injured  people"
    
    system_guidance = sector_messages.get(sector, "answer without emojis ,")

    dialogs = [
        [{"role": "system", "content": system_guidance }, {"role": "user", "content": prompt_base}]
    ]

    return f"{dialogs}"


generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

res = translate(generate_text(create_dialogue(crisis="train accident in desouk-kafr esh-sheikh road",sector="health",is_injuries=1)), tgt_lang="arabic")
print(res[0]["generated_text"])

