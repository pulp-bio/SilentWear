import os
"""
Script to test models
"""

from pathlib import Path
ARTIFACTS_DIR = Path(os.environ.get('SILENTWEAR_ARTIFACTS_DIR', Path(__file__).resolve().parents[2]/'artifacts'))
import sys
import os
import pandas as pd
import json
import torch
import copy
import ast
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from utils.general_utils import load_all_h5files_from_folder
from models.TorchTrainer import TorchTrainer
from offline_experiments.Model_Master import Model_Master
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


fig_path = ARTIFACTS_DIR
def return_data_directories(main_data_dire_proc, sub_ids, all_subject_models, condition, win_size_ms):
    data_dire_proc = []
    if all_subject_models == False:
        # check if we want to train with silent - vocalized or both
        if condition != "voc_and_silent":
            data_dire_proc.append(main_data_dire_proc / Path(f"wins_and_feats/{sub_ids}/{condition}/WIN_{win_size_ms}"))
        else:
            data_dire_proc.append(main_data_dire_proc / Path(f"wins_and_feats/{sub_ids}/silent/WIN_{win_size_ms}"))
            data_dire_proc.append(main_data_dire_proc / Path(f"wins_and_feats/{sub_ids}/vocalized/WIN_{win_size_ms}"))

    elif all_subject_models == True:
        for curr_sub_id in sub_ids:
            if condition != "voc_and_silent":
                data_dire_proc.append(main_data_dire_proc / Path(f"win_and_feats/{curr_sub_id}/{condition}/WIN_{win_size_ms}"))
            else:
                data_dire_proc.append(main_data_dire_proc / Path(f"wins_and_feats/{curr_sub_id}/silent/WIN_{win_size_ms}"))
                data_dire_proc.append(main_data_dire_proc / Path(f"wins_and_feats/{curr_sub_id}/vocalized/WIN_{win_size_ms}"))
    for curr_data_dire_proc in data_dire_proc:        
        if curr_data_dire_proc.exists() == False:
            print("Data directory:", curr_data_dire_proc, "does not exist, exist")
            sys.exit()
    return data_dire_proc


if __name__=='__main__':
    
    #main_data_dire_proc = PROJECT_ROOT / 'data'
    main_data_dire_proc = Path("/baltic/users/ml_datasets/iis_bio_internal_datasets/2026_spacone_speech_classification_hmi")
    ########### LOAD THE MODEL #########################
    model_base_folder = r"/scratch2/gspacone/sensors_2026_final/models/inter_session_win_sweep/S02/vocalized/speechnet_padded/model_4"
    model_base_folder = PROJECT_ROOT / model_base_folder
    run_cfg_file = model_base_folder/'run_cfg.json'
    
    with open(run_cfg_file, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    cv_summary = pd.read_csv(model_base_folder/"cv_summary.csv")
    # Get model-related info
    model_cfg = run_cfg['model_cfg']['model']
    model_name = model_cfg['name']
    print("Model was:", model_name)
    # Get info
    subjects = run_cfg["base_cfg"]["data"]["subject_id"]
    if type(subjects) == list:
        all_subject_models = True
    elif type(subjects) == str:
        all_subject_models = False
    condition = run_cfg.get("condition", "silent")
    print("Model was trained on:", condition, "data" )
    win_size_ms = run_cfg.get("experimental_settings",{}).get("window_size_ms", "1000")
    print("Window size was:", win_size_ms, "ms")


    ############### LOAD DATA
    data_directories = return_data_directories(main_data_dire_proc=main_data_dire_proc, 
                                            sub_ids=subjects, all_subject_models=all_subject_models, 
                                            condition=condition, win_size_ms=win_size_ms)

    df = pd.DataFrame()
    for curr_data_dire in data_directories:
        df_curr = load_all_h5files_from_folder(curr_data_dire, key='wins_feats', print_statistics = False)
        df = pd.concat((df, df_curr))
        
    print("=================================================\n")
    for fold_id in range(1, 4):
    
        model_to_consider = f"leave_one_session_out_fold_{fold_id}.pt"
        model_master = Model_Master(base_config=run_cfg["base_cfg"],model_config=run_cfg["model_cfg"])

        requested_fold = fold_id
        # The "model_summary_file" contains predictions from the model 
        
        summary_current_fold = cv_summary[cv_summary['fold_num'] == requested_fold]
        test_session = summary_current_fold['test_session'].values
        if len(test_session)!=1:
            print("not supported")
        else:
            test_session = test_session[0]
        print("Model was tested on session:", test_session)

        ############## LOAD DATA ###############

        print(df['session_id'].unique())
        df_test = df[df['session_id'] == test_session]

        #df_test = df_test[df_test["batch_id"]==1]

        # Assing df test to model master

        model_master.df_test  = df_test
        

        model_master.generate_training_labels()
        words = list(model_master.train_label_map.items())
        model_master.remap_all_datasets()
        df_col = model_master.extract_dataset_train_columns()
        df_col = df_col + ['Label_train']
        model_master.register_model()


        # now, load weights
        cpt = torch.load(model_base_folder/model_to_consider, weights_only=False)
        # Save parameters BEFORE loading
        params_before = copy.deepcopy({k: v.clone() for k, v in model_master.model.state_dict().items()})
        # Load checkpoint
        model_master.model.load_state_dict(cpt["model_state_dict"], strict=True)

        # # Compare AFTER loading
        # changed = False
        # for k in params_before:
        #     if not torch.equal(params_before[k], model_master.model.state_dict()[k]):
        #         changed = True
        #         print(f"Parameter changed: {k}")
        #         break

        # print("Weights updated?" , changed)


        model_master.trainer_manager.test_loader = model_master.trainer_manager.create_dataloader_from_df(model_master.df_test[df_col], batch_size = 1, 
                                                                                                          shuffle = False, num_workers=0)
        metrics, y_true, y_pred = model_master.trainer_manager.evaluate()

        print(summary_current_fold)

        """
        #cm_1= np.asarray(ast.literal_eval(res.iloc[0]['confusion_matrix']))
        cm_saved = np.asarray(ast.literal_eval(summary_current_fold.iloc[0]["confusion_matrix"]))
        cm_new= np.asarray(metrics["confusion_matrix"])
        y_true = np.array(y_true)
        #print(y_true[:30])
        original = np.array(summary_current_fold["y_true"].values)
        #print(original[:30])


        fig, axs = plt.subplots(1, 2, figsize=(15,10))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_saved,display_labels=words)
        disp1.plot(ax=axs[0], cmap=plt.cm.Blues, colorbar=False)
        axs[0].set_title("Saved to csv")

        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_new,display_labels=words)
        disp2.plot(ax=axs[1], cmap=plt.cm.Blues, colorbar=False)
        axs[1].set_title("New")


        plt.savefig(f"{fig_path}/fold{fold_id}.png")
        """

    ######## UNCOMMENT TO GET MODEL INFO (params)
    
    # from torchinfo import summary
    # model = model_master.trainer_manager.model
    # device = next(model.parameters()).device
    # model.eval()
    # for inputs, targets in model_master.trainer_manager.test_loader:
    #     inputs = inputs.to(device)
    #     targets = targets.to(device)
    #     outputs = model(inputs)             # logits: (batch_size, num_classes)

    # # (Optional but recommended) remove hooks/side-effects and avoid double compute:
    # with torch.no_grad():
    #     _ = model(inputs)

    # # Correct torchinfo call:
    # summary(model, input_data=inputs)

    # for name, m in model.named_modules():
    #     if isinstance(m, torch.nn.Conv2d):
    #         print(name, "in", m.in_channels, "out", m.out_channels,
    #             "kernel", m.kernel_size, "stride", m.stride, "padding", m.padding)
            
       

