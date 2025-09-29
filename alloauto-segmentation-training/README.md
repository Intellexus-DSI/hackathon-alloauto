Code Switching to detenct ALLO/AUTO part of text

- (continue pre-trained model by 'OMRIDRORI/mbert-tibetan-continual-wylie-final')
- 4-class token classification: Auto, Allo, Switch→Auto, Switch→Allo
- Proximity-aware loss function with 5-token tolerance for switch boundaries
- The data will be created according to hackathon-alloauto/dataset/annotated-data-raw


To run fine tune script: 
from root dir: hackathon-alloauto run:

python alloauto-segmentation-training/fine_tune_CS_4_classes_clean_no_allo_auto_labels.py

this uses GPU if available
this script will puchlish the model to HuggingFace (currrently by name: levshechter/tibetan-CS-detector_mbert-tibetan-continual-wylie_all_data.
Can Be changed, need huggingface accesstoken - use: export HF_TOKEN=<your token> 


to evaluate with Mikes binay allo/auto setnetce level model vs the ALTO BeRT fine tuned model:
run also from hackathon-alloauto:


 python alloauto-segmentation-training/evaluate_alloauo_ft_CS_model_vs_Mik_model_fbeta.py


for inference (for presentation UI integration)

python alloauto-segmentation-training/inference_fine_tuned_CS_v2.py


for data preprocess (data that is .txt or .docx file with <auto> and <allo> in it -> to segments with 4 classes labels):
the preprocess is inside the fine tune script

