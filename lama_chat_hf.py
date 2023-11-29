from torch import cuda, bfloat16
import transformers

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
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

def create_dialouge(crisis, sector,is_injuries=0):
    crisis=translate_text(crisis, "english")
    sector=translate_text(sector, "english")
    if is_injuries:
        prompt = f"As an Egyptian political man Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis} Taking into account the presence of dead and injured  people, formatted as one action per line ending with a '+'."
    else:
        prompt = f"As an Egyptian political man Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis}, formatted as one action per line ending with a '+'."
    
    # create system guidance
    system_guidance= f"act as an Egyptian political man without emojis."

    if sector == "health":
        system_guidance =  f"It is imperative to consistently make decisions that safeguard the health of citizens, mitigate the risk of potential natural disasters, and offer recommendations aimed at preventing such occurrences. So, give me a short answer about {crisis} without emojis."
    elif sector == "national security":
        system_guidance =  f"Your decisions should be geared towards ensuring Egyptian national security and the safety of the people, all while respecting international and diplomatic boundaries. So, give me a short answer about {crisis} without emojis."
    elif sector == "economic":
        system_guidance =  f"You are required to make decisions that serve the best interests of the nation's economy, considering prevailing economic conditions, the stock market, and the guidance of the Central Bank of Egypt. Give a short answer about {crisis} without emojis."
    elif sector == "education":
        system_guidance =f" You are tasked with making decisions that prioritize the best interests of the students, fostering their learning experiences, and offering recommendations that contribute to the advancement of education. So, give me a short answer about {crisis} without emojis."
    elif sector == "foreign policy":
        system_guidance = f" The decision must align accurately with the policies and laws of the Arab Republic of Egypt, ensuring it is in the best interest of the country, all the while upholding strong international diplomatic relations. So, give me a short answer about {crisis} without emojis."
    elif sector == "media":
        system_guidance = f" Compliance with the regulations set forth by the unions associated with this sector is imperative, encompassing both audio-visual and written domains. So, give me a short answer about {crisis} without emojis."


    # create dialog
    
    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": system_guidance },  
            {"role": "user", "content": prompt}
        ]
    ]

    return dialogs


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

res = generate_text("Explain to me the difference between Arabic union and Urobian union ?")
print(res[0]["generated_text"])

