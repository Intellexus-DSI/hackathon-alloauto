from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import os
import json
import yaml
import json
import logging
from typing import List, Generator

def get_llm(temperature=0.7, model_name="gemini-2.5-flash"):
    """Initialize LLM with specified parameters"""
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )

#=======LONG PROMPT=======#
# PROMPT = {
#     "system": """You are a Tibetan Buddhist philology expert and computational linguist. Read Tibetan text and segment it into contiguous spans of two types:
#     AUTO (autochthonous Tibetan): passages originally composed in Tibetan.
#     ALLO (allochthonous Tibetan): passages translated into Tibetan (typically from Sanskrit or related Indic sources).
#     Output: Return only JSON of the form: "prediction": [i1, i2, ...]
#     Each integer is a 0‑based word index that marks the start of a new segment after a label change.
#     You must output at least one index (best-guess boundary). Never return an empty list.
#     Word indexing (basis for indices):
#     - Normalize consecutive whitespace to single spaces for counting.
#     - The first word has index 0.
#     - Do not include index 0 in prediction.
#     - Indices must be integers, unique, ascending, each within [1, W-1] for total words W.
#     Minimal segmentation guidance (to avoid empty outputs):
#     - Choose a starting label for the first span (default AUTO) and scan left→right.
#     - Switch to ALLO when you encounter sustained translationese cues, or a strong hard marker. Switch back to AUTO when those cues subside for a similar stretch.
#     - If uncertain in a long passage (≥150 words), make at least one best‑guess switch at the most probable boundary rather than returning an empty list.
#     Common ALLO cues (examples—not exhaustive):
#     - Dhāraṇī/mantra blocks or seed syllables: ཨོཾ, ཧཱུྃ, ཙྪཿ, repeated invocatory formulae; long phonetic strings with minimal Tibetan particles.
#     - Dense Indic proper names/transliterations and non‑Tibetan orthography bursts (e.g., stacked consonant transliterations, visarga‑like signs, siddham marks).
#     - Literal scholastic calque feel: extended chains mirroring Sanskrit compounds and rigid enumerations; repetitive frames akin to namo, iti, evaṃ mayā śrutam equivalents.
#     - Function word profile shift: unusually sparse use of idiomatic Tibetan connective particles relative to long technical noun compounds.
#     Common AUTO cues:
#     - Idiomatic expository/narrative flow, local examples, freer paraphrase, pragmatic asides.
#     - Tibetan rhetorical turns and connective particles used in natural distribution.
#     Hard markers that justify an immediate switch (even within <12 words):
#     - Start/end of mantra/dhāraṇī section.
#     - A contiguous run of transliteration‑like syllables or non‑Tibetan letter clusters.
#     Tie‑breaking & stability:
#     - When signals are mixed, maintain the current label until evidence accumulates; but avoid returning an empty list for long mixed passages—choose the most plausible single boundary.
#     Formatting constraints:
#     - Output only the JSON object.
#     - Do not include any explanation, counts, or labels.
# """,
#     "user": "Text: {text}\n"
# }
#=======LONG PROMPT=======#

#=======SHORT PROMPT=======#
PROMPT = {
    "system": """
    You are a Tibetan Buddhist philology expert and computational linguist. Your task is to read Tibetan text and segment it into contiguous spans of two types:
    AUTO (autochthonous Tibetan): passages originally composed in Tibetan. ALLO (allochthonous Tibetan): passages translated into Tibetan (typically from Sanskrit or related Indic sources).
    Return only JSON of the form: ("first_segment": allo/auto,"prediction": [i1, i2, ...]) Each integer is a 0‑based word index that marks the start of a new segment after a label change. If there is no switch, return "prediction": [].
    Split the given text into words as follows:
    - Normalize consecutive whitespace to single spaces for counting.
    - first_segment is the first segment of the text, either "allo" or "auto".
    - The first word has index 0.
    - Do not include index 0 in prediction.
    - Indices must be integers, unique, sorted ascending, each within [1, W-1] for W total words.
    Output only the JSON object; no explanations or extra text.
    """,
    "user": "Text: {text}\n"
}

PROMPT_ZERO_SHOT =  ChatPromptTemplate.from_messages([
    ("system", PROMPT["system"]),
    ("user", PROMPT["user"])
])

# === FEW-SHOT PROMPT ===
# Define your Sanskrit examples with CORRECTED output format
examples = [
    # Example 1: 
    # {
    #     "input":  "byams pa chen po rgyun chad pa nyan thos pa la dgag bya'i gtso bo ma yin pa'i phyir ro / mu gsum yongs su dag pa 'di ni ngas rim gyis bslab pa'i gzhi bsngam pa'i phyir zhes sogs kyis kyang / nyan thos pas sha za ba bkag pa ma yin te / dper na / bstan pa la rim gyis gzhug pa'i phyir du / theg pa gsum gsungs pas nyan thos pa la rang don don gnyer gyi bsam pa ma bkag pa bzhin no / des na mdo de dag gis byang sems las dang po pa sha la sred pa'i @# / dbang gis byams pa chen po rgyun chad par 'gyur ba la sha sred pas za ba bkag pa yin te / sha za ba ni byams pa chen po chad par 'gyur ro / zhes dang / ngas ni lus g. yog pa'i phyir gos sna tshogs kyang kha dog ngan par bsgyur bar bya'o zhes bstan na / sha za ba'i ro la chags pa lta ci smos / zhes sogs dang / rgyu de dag gis na byang chub sems dpa' sems dpa' chen po rnams sha mi za'o / zhes gsungs pa'i phyir ro / nga'i nyan thos rnams sha za bar mi gnang ngo zhes pas kyang mi gnod de / byang chub tu bgrod pa'i lam sangs rgyas las nyan nas / don de gzhan la thos par byed pas na / nyan thos kyi sgra byang sems la yang bshad du yong pa'i phyir ro / gal te 'dul ba nas rab byung gis rnam gsum dag pa'i sha za ba gnang yang / phyis mdo gzhan nas rab byung gis sha za ba bkag pas / 'dul ba nas rab byung gis sha za bar gnang ba ni /",
    #     "output": '{"prediction": [130, 183]}'
    # },
    # # Example 2: 
    # {
    #     "input": "vratāya tenānucareṇa dhenor nyaṣedhi śeṣo 'py anuyāyivargaḥ |\nna cānyatas tasya śarīrarakṣā svavīryaguptā hi manoḥ prasūtiḥ \n|| 4 || \n\nvratāye\nti | rājñā devī na kevalaṃ nyaṣedhi\n \nkintu \ndhenor\n \nanucareṇa\n \ntena śeṣo ’pi anuyāyivargaḥ\n \nnyaṣedhi\n nagarasthāpitāpekṣayā śeṣatvam | svadeharakṣaṇārthaṃ kecit kuto na rakṣitā ity āha—yatas \ntasya anyataḥ śarīrarakṣā na\niva syur evaṃ tu punar vaivety avadhāraṇavācakāḥ- kuta ity āha— \nhi \nyasmāt\nkāraṇāt \nmanoḥ prasūtiḥ\n prasūyate iti prasūtiḥ santatiḥ \nsvavīryaguptā\n svavīryeṇaiva rakṣitā | na hi svaparanirvāhakasya parāpekṣeti bhāvaḥ || 4 ||",
    #     "output": '{"prediction": [["VERSE", "vratāya tenānucareṇa dhenor nyaṣedhi śeṣo py anuyāyivargaḥ |\nna cānyatas tasya śarīrarakṣā svavīryaguptā hi manoḥ prasūtiḥ \n||"], ["INCOMPLETE_VERSE", "syur evaṃ tu punar vaivety avadhāraṇavācakāḥ"]]}'
    # },
    # # Example 3: 
    # {
    #     "input": "āsvādavadbhiḥ kavalais tṛṇānāṃ kaṇḍūyanair daśanivāraṇaiś ca |\navyāhataiḥ svairagataiḥ sa tasyāḥ samrāṭ samārādhanatatparo 'bhūt\n || 5 ||\n\nāsvāde\nti | \nsa samrāṭ\n yeneṣṭaṃ rājasūyena maṇḍalasyeśvaraś ca yaḥ śāsti yaś cājñayā rājñaḥ sa samrāṭ parikīrtitaḥ mo rāji samaḥ kvau \ntasyā\n dhenoḥ sevā\nparo ’bhūt\n | kaiḥ āsvāda eṣām astīti āsvādavantaḥ taiḥ \ntṛṇānāṃ\n grāsaiḥ grāsas tu kavala kaiḥ \nkaṇḍūyanaiḥ\n kaṣaṇaiś ca \ndaṃśanivāraṇaiś ca\n | daṃśas tu vanamakṣikā na vyāhatāni svairāṇi ca tāni gatāni ca taiḥ | mandasvacchandayoḥ svairam || 5 ||",
    #     "output": '{"prediction": [["VERSE", "āsvādavadbhiḥ kavalais tṛṇānāṃ kaṇḍūyanair daśanivāraṇaiś ca |\navyāhataiḥ svairagataiḥ sa tasyāḥ samrāṭ samārādhanatatparo bhūt"], ["VERSE", "yeneṣṭaṃ rājasūyena maṇḍalasyeśvaraś ca yaḥ śāsti yaś cājñayā rājñaḥ sa samrāṭ parikīrtitaḥ"], ["INCOMPLETE_VERSE", "grāsas tu kavala"], ["INCOMPLETE_VERSE", "daṃśas tu vanamakṣikā"], ["INCOMPLETE_VERSE", "mandasvacchandayoḥ svairam"]]}'
    # }

]

# Define how each example should be formatted
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("assistant", "{output}")
])

# Create the few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Reuse the zero-shot prompt and add few-shot examples
PROMPT_FEW_SHOT = ChatPromptTemplate.from_messages([
    *PROMPT_ZERO_SHOT.messages[:-1],  # All messages except the last user message
    few_shot_prompt,
    PROMPT_ZERO_SHOT.messages[-1]     # The original user message template
])

# === CONFIGURATION-BASED PROMPT SELECTION ===

def get_prompt(use_few_shot=False):
    """
    Get the appropriate prompt template based on configuration.
    Args:
        use_few_shot (bool): If True, use few-shot prompting. If False, use simple prompt.   
    Returns:
        ChatPromptTemplate: The selected prompt template
    """
    if use_few_shot:
        print("Using Few-Shot Prompting approach")
        return PROMPT_FEW_SHOT
    else:
        print("Using zero-shot Prompting approach")
        return PROMPT_ZERO_SHOT


def set_env_vars(config_path="keys.yaml"):
    """Set OS env variables from a YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
    with open(config_path, "r") as f:
        keys = yaml.safe_load(f)
        for key, value in keys.items():
            os.environ[key] = str(value)

def save_results(results, output_file):
    """Save results to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}
    
        
def set_logger(logger: logging.Logger, full_msg: bool = False) -> logging.Logger:
    """Set logger to report file"""
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    if full_msg:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    else:
        fmt = logging.Formatter("%(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger

def write_report(messages:List[str], report_file: str) -> None:
    """Write messages to report file"""
    with open(report_file, "w", encoding="utf-8") as f:
        for message in messages:
            f.write(message + "\n")

def load_results_json(file_path: str) -> Generator[dict, None, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            yield {
                'first_segment': json_line['first_segment'],
                'predictions': json_line['predictions'],
                'total_tokens': json_line['total_tokens'],
                'labeled_array': json_line['labeled_array'],
                'sample_id': json_line['sample_id'],
            }
            
def fill_class_segments(results: List[dict]) -> List[List[int]]:
    """Fill class segments in results"""
    full_predictions = []
    for result in results:
        first_segment = result['first_segment']
        segment_breaks = list(result['predictions'])
        total_tokens = result['total_tokens']

        if first_segment == 'auto':
            curr_class, curr_switch_class = 0, 3
        elif first_segment == 'allo':
            curr_class, curr_switch_class = 1, 2
        else:
            raise ValueError(f"Invalid first segment: {first_segment}")

        if segment_breaks == []:
            full_predictions.append([curr_class] * total_tokens)
            continue

        prev_break_idx = 0
        curr_labels = [None] * total_tokens
        for break_idx in segment_breaks:
            for i in range(prev_break_idx, break_idx):
                curr_labels[i] = curr_class
            curr_labels[break_idx] = curr_switch_class

            curr_class = 0 if curr_class == 1 else 1
            curr_switch_class = 2 if curr_switch_class == 3 else 3
            prev_break_idx = break_idx + 1

        for i in range(prev_break_idx, total_tokens):
            curr_labels[i] = curr_class

        full_predictions.append(curr_labels)
    return full_predictions
    
def convert_to_labeled_array(prediction: List[int], first_segment: str, total_tokens: int) -> List[int]:
    """Convert single prediction to array"""
    if first_segment == 'auto':
        curr_class, curr_switch_class = 0, 3
    elif first_segment == 'allo':
        curr_class, curr_switch_class = 1, 2
    else:
        raise ValueError(f"Invalid first segment: {first_segment}")

    if prediction == []:
        return [curr_class] * total_tokens

    prev_break_idx = 0
    labels = [None] * total_tokens
    for break_idx in prediction:
        for i in range(prev_break_idx, break_idx):
            labels[i] = curr_class
        labels[break_idx] = curr_switch_class

        curr_class = 0 if curr_class == 1 else 1
        curr_switch_class = 2 if curr_switch_class == 3 else 3
        prev_break_idx = break_idx + 1

    for i in range(prev_break_idx, total_tokens):
        labels[i] = curr_class

    return labels

def get_closed_models_predictions(file_path: str) -> List[int]:
    """Get metrics for closed models predictions"""
    results = load_results_json(file_path)
    predictions = []
    for result in results:
        predictions.extend(result['labeled_array'])
    return predictions
