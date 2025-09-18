Code Switching to detenct ALLO/AUTO part of text

- (continue pre-trained model by 'OMRIDRORI/mbert-tibetan-continual-wylie-final')
- 4-class token classification: Auto, Allo, Switch→Auto, Switch→Allo
- Proximity-aware loss function with 5-token tolerance for switch boundaries
- The data will be created according to hackathon-alloauto/dataset/all data


To run fine tune script: 
from root dir: hackathon-alloauto run:

python classify_allo_auto/classify_allo_auto/fine_tune_CS_4_classes_clean_no_allo_auto_labels.py

this uses GPU if available
this script will puchlish the model to HuggingFace (currrently by name: levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_all_data.
Can Be changed, need huggingface accesstoken - use: export HF_TOKEN=<your token> 


to eva;uate bith Mikes binay allo/auto setnetce level model vs the ALTO BeRT fine tuned model:
run also from hackathon-alloauto:


 python classify_allo_auto/evaluate_alloauo_ft_CS_model_vs_Mik_model_fbeta.py
