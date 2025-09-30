import os
import warnings
import random
from pathlib import Path
import traceback
import utils
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import logging

# Suppress Google Cloud warnings  
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="google.*")

# Logger for console output
logger = logging.getLogger(__name__)
logger = utils.set_logger(logger, full_msg=False)

# Messages for report
messages = []
def _append_msg(msg: str) -> None:
    if not msg:
        return
    messages.append(msg)

def main():
    # Load API keys first
    try:
        utils.set_env_vars("keys.yaml")
        _append_msg("🔑 API keys loaded from keys.yaml")
        logger.info("🔑 API keys loaded from keys.yaml")
    except:
        logger.warning("⚠️ No keys.yaml found, assuming API keys are set")
    
    # Load configuration
    config = utils.load_config()
    
    # Set random seed for reproducibility
    random.seed(config.get('seed', 42))
    
    # Log configuration
    _append_msg("=" * 50)
    _append_msg("VERSE EXTRACTION - Configuration")
    _append_msg("=" * 50)

    for key, value in config.items():
        _append_msg(f"  {key}: {value}")
    _append_msg("=" * 50)
    
    # Check if data file exists
    DATA_DIR = config.get('data_dir', '')
    DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR)
    if not DATA_DIR or not Path(DATA_DIR).exists():
        logger.error(f"❌ Error: Data dir not found: {DATA_DIR}")
        return    

    # Initialize the LLM with config
    logger.info("🔄 Initializing LLM")
    try:
        llm = utils.get_llm(
            temperature=config.get('temperature', 0.3),
            model_name=config.get('model_name', "gemini-2.5-flash")
        )
        _append_msg(f"✅ LLM initialized: {config.get('model_name', 'gemini-2.5-flash')}")
        logger.info(f"✅ LLM initialized: {config.get('model_name', 'gemini-2.5-flash')}")
    except Exception as e:
        logger.error(f"❌ Error initializing LLM: {e}")
        return
    
    # Load data
    DATA_FILE_NAME = config.get('data_file_name', '')
    logger.info(f"🔄 Loading data from {os.path.join(DATA_DIR, DATA_FILE_NAME)}")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE_NAME))
        _append_msg(f"✅ Loaded {len(df)} samples from {os.path.join(DATA_DIR, DATA_FILE_NAME)}")
        logger.info(f"✅ Loaded {len(df)} samples from {os.path.join(DATA_DIR, DATA_FILE_NAME)}")
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return
    
    samples = df["original_text"].tolist()

    # Apply debug sampling if specified
    debug_samples = config.get('debug_samples', 0)
    if debug_samples > 0:
        samples = samples[:debug_samples]
        _append_msg(f"🐛 Debug mode: processing only {len(samples)} samples")
        logger.info(f"🐛 Debug mode: processing only {len(samples)} samples")
    
    # Get the appropriate prompt
    use_few_shot = config.get('few_shot', True)
    prompt = utils.get_prompt(use_few_shot=use_few_shot)
    _append_msg(f'prompt:\nprompt={prompt}')

    # Create chain
    chain = prompt | llm
    
    _append_msg("=" * 50)

    # Process samples
    results = []
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc="Processing samples"):
        input_text = sample
        if not input_text:
            _append_msg(f"⚠️  Skipping sample {i+1}: No text found")
            logger.warning(f"⚠️  Skipping sample {i+1}: No text found")
            continue
        
        _append_msg(f"🔄 Processing sample {i+1}")
        _append_msg(f"📄 Input length: {len(input_text)} characters")
        
        try:
            response = chain.invoke({"text": input_text})
            response_content = response.content if hasattr(response, 'content') else str(response)
            cleaned_response = response_content.strip('`').replace('json\n', '')
            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response: {cleaned_response}")

            _append_msg(f"🔍 Usage metadata:\n{response.usage_metadata}")

            predictions = result["prediction"] if "prediction" in result else []
        
            results.append({
                'predictions': predictions,
                'result': result,
                'sample_id': i+1,
                'approach': 'few-shot' if use_few_shot else 'simple',
            })

            _append_msg(f"✅ Sample {i+1} processed successfully")
            _append_msg(f"📊 Result: {result}")
            if debug_samples > 0:
                tqdm.write(f"✅ Sample {i+1} processed successfully")
                tqdm.write(f"📊 Result: {result}")

        except Exception as e:
            _append_msg(f"❌ Error processing sample {i+1}: {e}")
            logger.error(f"❌ Error processing sample {i+1}: {e}")
            logger.error(traceback.format_exc())
            results.append({
                'predictions': predictions,
                'sample_id': i+1,
                'input_text': input_text,
                'result': None,
                'error': str(e),
                'approach': 'few-shot' if use_few_shot else 'zero-shot',
            })
        
        _append_msg("-" * 30)
    
    # Save results
    try:
        approach_name = "few_shot" if use_few_shot else "zero_shot"
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        output_filename = f"results_{approach_name}_{len(results)}_samples.jsonl"
        output_file = os.path.join(results_dir, output_filename)
        utils.save_results(results, output_file)
        _append_msg(f"💾 Results saved to: {output_file}")
        logger.info(f"💾 Results saved to: {output_file}")
    except Exception as e:
        _append_msg(f"❌ Error saving results: {e}")
        logger.error(f"❌ Error saving results: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("=" * 50)
    logger.info("📈 Writing report")
    utils.write_report(messages, "./closed models/report.log")
    logger.info("✅ Report written to: report.log")
    # Summary
    # print("=" * 50)
    # print("📈 SUMMARY")
    # print("=" * 50)

    # ADD THIS NEW SECTION:
    # Calculate and display character-level metrics

if __name__ == "__main__":
    main()