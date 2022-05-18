# %%
import os
# %%
# pre-upsample model
if not os.path.exists("../Data-Pre-upsample/10x_train_syn_B"):
    os.makedirs("../Data-Pre-upsample/10x_train_syn_B")
if not os.path.exists("../Data-Pre-upsample/10x_train_syn_S"):
    os.makedirs("../Data-Pre-upsample/10x_train_syn_S")
if not os.path.exists("../Data-Pre-upsample/10x_eval_syn_B"):
    os.makedirs("../Data-Pre-upsample/10x_eval_syn_B")
if not os.path.exists("../Data-Pre-upsample/10x_eval_syn_S"):
    os.makedirs("../Data-Pre-upsample/10x_eval_syn_S")

if not os.path.exists("../Data-Pre-upsample/20x_train_syn_B"):
    os.makedirs("../Data-Pre-upsample/20x_train_syn_B")
if not os.path.exists("../Data-Pre-upsample/20x_train_syn_S"):
    os.makedirs("../Data-Pre-upsample/20x_train_syn_S")
if not os.path.exists("../Data-Pre-upsample/20x_eval_syn_B"):
    os.makedirs("../Data-Pre-upsample/20x_eval_syn_B")
if not os.path.exists("../Data-Pre-upsample/20x_eval_syn_S"):
    os.makedirs("../Data-Pre-upsample/20x_eval_syn_S")

if not os.path.exists("../Data-Pre-upsample/10x_predict"):
    os.makedirs("../Data-Pre-upsample/10x_predict")
if not os.path.exists("../Data-Pre-upsample/20x_truth"):
    os.makedirs("../Data-Pre-upsample/20x_truth")
