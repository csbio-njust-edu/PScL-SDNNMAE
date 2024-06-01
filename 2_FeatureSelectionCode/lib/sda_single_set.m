function sda_single_set(read_dir_path, write_dir_path)
clbp_val_path = read_dir_path + "/am/val/CLbp_am.mat";
lbp_val_path = read_dir_path + "/am/val/LBP_am.mat";
let_val_path = read_dir_path + "/am/val/LET_am.mat";
riclbp_val_path = read_dir_path + "/am/val/RICLBP_am.mat";
slfs_val_path = read_dir_path + "/am/val/SLFs_am.mat";
mae_val_path = read_dir_path + "/am/val/MAE_am.mat";
label_val_path = read_dir_path + "/am/val/label.mat";

CLBP_test =  load(clbp_val_path).CLBP;
LBP_test = load(lbp_val_path).LBP;
LET_test = load(let_val_path).LET;
RICLBP_test = load(riclbp_val_path).RICLBP;
SLFs_test = load(slfs_val_path).SLFs;
MAE_test = load(mae_val_path).MAE;
label_test = load(label_val_path).label;

clbp_train_path = read_dir_path + "/am/train/CLbp_am.mat";
lbp_train_path = read_dir_path + "/am/train/LBP_am.mat";
let_train_path = read_dir_path + "/am/train/LET_am.mat";
riclbp_train_path = read_dir_path + "/am/train/RICLBP_am.mat";
slfs_train_path = read_dir_path + "/am/train/SLFs_am.mat";
mae_train_path = read_dir_path + "/am/train/MAE_am.mat";
label_train_path = read_dir_path + "/am/train/label.mat";

CLBP = load(clbp_train_path).CLBP;
LBP = load(lbp_train_path).LBP;
LET = load(let_train_path).LET;
RICLBP = load(riclbp_train_path).RICLBP;
SLFs = load(slfs_train_path).SLFs;
MAE = load(mae_train_path).MAE;
label = load(label_train_path).label;

CLBP = featnorm(CLBP);
CLBP_test = featnorm(CLBP_test);
[idx sda_CLBP sda_CLBP_test] = SDA_FeatSelect(CLBP, CLBP_test, label);
save(write_dir_path + "/sda/train/CLBP_sda.mat", "sda_CLBP");
save(write_dir_path + "/sda/val/CLBP_sda.mat", "sda_CLBP_test");

LBP = featnorm(LBP);
LBP_test = featnorm(LBP_test);
[idx sda_LBP sda_LBP_test] = SDA_FeatSelect(LBP, LBP_test, label);
save(write_dir_path + "/sda/train/LBP_sda.mat", "sda_LBP");
save(write_dir_path + "/sda/val/LBP_sda.mat", "sda_LBP_test");

RICLBP = featnorm(RICLBP);
RICLBP_test = featnorm(RICLBP_test);
[idx sda_RICLBP sda_RICLBP_test] = SDA_FeatSelect(RICLBP, RICLBP_test, label);
save(write_dir_path + "/sda/train/RICLBP_sda.mat", "sda_RICLBP");
save(write_dir_path + "/sda/val/RICLBP_sda.mat", "sda_RICLBP_test");

LET = featnorm(LET);
LET_test = featnorm(LET_test);
[idx sda_LET sda_LET_test] = SDA_FeatSelect(LET, LET_test, label);
save(write_dir_path + "/sda/train/LET_sda.mat", "sda_LET");
save(write_dir_path + "/sda/val/LET_sda.mat", "sda_LET_test");

SLFs = featnorm(SLFs);
SLFs_test = featnorm(SLFs_test);
[idx sda_SLFs sda_SLFs_test] = SDA_FeatSelect(SLFs, SLFs_test, label);
save(write_dir_path + "/sda/train/SLFs_sda.mat", "sda_SLFs");
save(write_dir_path + "/sda/val/SLFs_sda.mat", "sda_SLFs_test");

MAE = featnorm(MAE);
MAE_test = featnorm(MAE_test);
[idx sda_MAE sda_MAE_test] = SDA_FeatSelect(MAE, MAE_test, label);
save(write_dir_path + "/sda/train/MAE_sda.mat", "sda_MAE");
save(write_dir_path + "/sda/val/MAE_sda.mat", "sda_MAE_test");

end

