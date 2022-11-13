import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import sys

DISEASE_NAMES = ["C0006142_Malignant_neoplasm_of_breast",  "C0009402_Colorectal_Carcinoma", "C0023893_Liver_Cirrhosis_Experimental","C0376358_Malignant_neoplasm_of_prostate","C0036341_Schizophrenia"] 
''',"C0009402_Colorectal_Carcinoma",  
    "C0036341_Schizophrenia", "C0001973_Alcoholic_Intoxication_Chronic", "C0011581_Depressive_disorder", "C0860207_Drug_Induced_Liver_Disease",
                 "C3714756_Intellectual_Disability", "C0005586_Bipolar_Disorder"]'''

DISEASE_CODES = {"C0006142_Malignant_neoplasm_of_breast": "C0006142", "C0009402_Colorectal_Carcinoma": "C0009402", "C0023893_Liver_Cirrhosis_Experimental": "C0023893",
    "C0036341_Schizophrenia": "C0036341", "C0376358_Malignant_neoplasm_of_prostate": "C0376358", "C0001973_Alcoholic_Intoxication_Chronic": "C0001973", "C0011581_Depressive_disorder": "C0011581", 
    "C0860207_Drug_Induced_Liver_Disease": "C0860207", "C3714756_Intellectual_Disability": "C3714756", "C0005586_Bipolar_Disorder": "C0005586"}


# COMPARE_METHODS = ["DIAMOnD", "MCL", "RWR", "fFlow", "NetCombo", "NetRank"]
# COMPARE_METHODS = ["DIAMOnD", "XGDAG", "XGDAG + LP", "fFlow", "NetCombo", "NetRank"]
GUILD_METHODS = ["fFlow", "NetScore", "NetZcore", "NetShort","NetCombo", "NetRank"]
# COMPARE_METHODS = ["DIAMOnD", "GNNExplainer", "XGDAG"]
COMPARE_METHODS = ["DIAMOnD", "GNNExplainer", "XGDAG - GNNExplainer", "GraphSVX", "XGDAG - GraphSVX", "MCL", "RWR", "fFlow", "NetCombo", "NetRank"]

XAI_METHODS = ["GNNExplainer", "XGDAG - GNNExplainer", "GraphSVX", "XGDAG - GraphSVX", "SubraphX", "XGDAG - SubgraphX",
                "EdgeSHAPer", "XGDAG - EdgeSHAPer"]

PLOT = True
SAVE_METRICS = False
SAVE_RANKING_GENES = False
extended_validation = True
# ratios_to_validate = [n/10 for n in range(1, 11)]
ratios_to_validate = [25, 50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000]
print(ratios_to_validate)
for DISEASE_NAME in tqdm(DISEASE_NAMES):
    
    
    recall_folds = []
    precision_folds = []
    F1_folds = []

    recall_folds_compare_methods = {}
    precision_folds_compare_methods = {}
    F1_folds_compare_methods = {}

    for METHOD in COMPARE_METHODS:
        recall_folds_compare_methods[METHOD] = []
        precision_folds_compare_methods[METHOD] = []
        F1_folds_compare_methods[METHOD] = []

    # recall_folds_compare_methods = {"DIAMOnD":[], "MCL": [], "RWR": [], "fFlow":[], "NetScore":[], "NetZcore":[], "NetShort": [], "NetCombo": [], "NetRank":[],}
    # precision_folds_compare_methods = {"DIAMOnD":[], "MCL": [], "RWR": [], "fFlow":[], "NetScore":[], "NetZcore":[], "NetShort": [], "NetCombo":[], "NetRank":[]}
    # F1_folds_compare_methods = {"DIAMOnD":[], "MCL": [], "RWR": [], "fFlow":[], "NetScore":[], "NetZcore":[], "NetShort": [], "NetCombo":[], "NetRank":[]}

    
    # recall_folds_compare_methods["XGDAG"] = []
    # precision_folds_compare_methods["XGDAG"] = []
    # F1_folds_compare_methods["XGDAG"] = []

    # recall_folds_compare_methods["GNNExplainer"] = []
    # precision_folds_compare_methods["GNNExplainer"] = []
    # F1_folds_compare_methods["GNNExplainer"] = []

    for ratio_to_validate in ratios_to_validate:
        
        
        GENE_APU_SCORES_PATH = "Rankings/other_methods/NIAPU/" + DISEASE_NAME + "/" + DISEASE_NAME + "_ranking"
        TRAIN_SEEDS_PATH = "Datasets_v2/" + DISEASE_CODES[DISEASE_NAME] + "_seed_genes.txt"

        APU_scores_df = pd.read_csv(GENE_APU_SCORES_PATH, header = None, sep = " ")
        APU_scores_df.columns = ["name", "score", "label"]
        #APU_scores_df['name'] = APU_scores_df['name'].str.replace("ORF",'orf')
        APU_ranking_df = APU_scores_df.sort_values(by = "score", ascending= False)
        

        train_seeds_df = pd.read_csv(TRAIN_SEEDS_PATH, header = None, sep = " ") #seed genes used for diffusion that we cosnider as P class in this scenario (20% or seed genes were removed to check for robustness)
        train_seeds_df.columns = ["name", "GDA Score"]
        train_seeds_list = train_seeds_df["name"].values.tolist()

        APU_ranking_df_not_seeds = APU_ranking_df[~APU_ranking_df['name'].isin(train_seeds_list)]

        APU_ranking_candidate_genes = APU_ranking_df_not_seeds["name"].values.tolist()
        
        N = None
        
        FILE_NAME_ALL_SEEDS = "Datasets_v2/all_seed_genes/" + DISEASE_NAME + "_all_seed_genes.txt"
        all_seed_genes_df = pd.read_csv(FILE_NAME_ALL_SEEDS, sep = " ", header = None)
        all_seed_genes = all_seed_genes_df[0].values
        test_seeds = list(set(all_seed_genes).difference(set(train_seeds_list)))

        if SAVE_RANKING_GENES:    
            with open("../APU_SCORES/APU_scores_C_implementation/APU_rankings_no_scores_no_seed_genes/" + DISEASE_NAME + "_NIAPU_ranking_no_scores.txt", "w+") as saveFile:
                for candidate_gene in APU_ranking_candidate_genes:
                    saveFile.write(candidate_gene + "\n")
        
        N = len(all_seed_genes.tolist())
        #N = len(train_seeds_list)
        ##@mastro this is the max number of genes that can be found by the aglorithm
        #N  = len(test_seeds)
        
        
        
        
        APU_ranking_candidate_genes = APU_ranking_candidate_genes[:round(ratio_to_validate)]
        # LP = LP[:50]
        # print("LP ", len(LP))
        #LP = LP[:len(test_seeds)]
        TP = 0
        FP = 0
        P = len(test_seeds) #TP+FP
        #P = len(LP)
        for gene in APU_ranking_candidate_genes:
            
            if gene in test_seeds:
                TP += 1
                
            else:
                FP += 1

        # P = TP+FP
        # print(len(test_seeds))
        # print(P)
        # print("TP", TP)
        # print("FP", FP)
        recall = TP / P
        precision = TP / (TP + FP)

        F1_score = 0
        if (precision + recall) != 0:
            F1_score = 2*(precision*recall)/(precision+recall)

        recall_folds.append(recall)
        precision_folds.append(precision)
        F1_folds.append(F1_score)

        for METHOD in COMPARE_METHODS:
            ranking_method = []

            if METHOD in XAI_METHODS:
                with open("Rankings/" + DISEASE_CODES[DISEASE_NAME] + "_all_positives_new_ranking_" + METHOD.lower().replace("-", "_").replace(" ", "") + ".txt", "r", encoding="utf-8") as rankingFile:
                    for line in rankingFile:
                        ranking_method.append(line.strip("\n"))

            elif METHOD in GUILD_METHODS:        
                GUILD_METHOD_PATH = "Rankings/other_methods/GUILD/" + METHOD + "/" + DISEASE_NAME + "_" + METHOD + ".txt"

                GUILD_scores_df = pd.read_csv(GUILD_METHOD_PATH, header = None, sep = "\t")
                GUILD_scores_df.columns = ["name", "score"]
                GUILD_scores_df = GUILD_scores_df.sort_values(by = "score", ascending= False)

                ranking_method_df_not_seeds = GUILD_scores_df[~GUILD_scores_df['name'].isin(train_seeds_list)]
                ranking_method = ranking_method_df_not_seeds["name"].values.tolist()
               
            else:
                with open("Rankings/other_methods/" + METHOD + "/" + METHOD.lower() + "_output_" + DISEASE_NAME + ".txt", "r", encoding="utf-8") as rankingFile:
                        for line in rankingFile:
                            ranking_method.append(line.strip("\n"))
                

            ranking_method = ranking_method[:round(ratio_to_validate)]
            TP = 0
            FP = 0
            P = len(test_seeds) #TP+FP
            #P = len(LP)
            for gene in ranking_method:
                
                if gene in test_seeds:
                    TP += 1
                    
                else:
                    FP += 1

            # P = TP+FP
            # print(len(test_seeds))
            # print(P)
            # print("TP", TP)
            # print("FP", FP)
            recall = TP / P
            precision = TP / (TP + FP)
            
            F1_score = 0
            if (precision + recall) != 0:
                F1_score = 2*(precision*recall)/(precision+recall)

            recall_folds_compare_methods[METHOD].append(recall)
            precision_folds_compare_methods[METHOD].append(precision)
            F1_folds_compare_methods[METHOD].append(F1_score)
        
                    


    if PLOT:
        
        ratios = [str(n) for n in ratios_to_validate]
        
        dis_name_full = DISEASE_NAME.split("_")
        dis_code = dis_name_full[0]
        dis_name = " ".join(dis_name_full[1:]).lower()
        dis_name_title = dis_code + " - " + dis_name
        
        # sns.lineplot(x=ratios, y=precision_folds)
        # plt.ylabel("Precision")
        # plt.xlabel("Number of candidate genes")
        # # plt.legend(["NIAPU P"])
        # plt.ylim(0, 1)
        # plt.xticks(ratios)
        # # plt.yticks(precision_folds)
        # plt.savefig("../results/gene_discovery/new_APU_labelling/precision_" + DISEASE_NAME + "_NIAPU.png")
        # plt.clf()

        #to plot recall and prec for NIAPU
        # sns.lineplot(x=ratios, y=precision_folds, label="NIAPU P")
        ##recall###
        max_rec = 0
        figure(figsize=(16, 10))
        sns.set(font_scale=1.5)
        
        # sns.lineplot(x=ratios, y=recall_folds_compare_methods["XGDAG"],label="XGDAG")
        # max_rec = max(recall_folds_compare_methods["XGDAG"])

        # sns.lineplot(x=ratios, y=recall_folds_compare_methods["XGDAG + LP"],label="XGDAG")
        # max_rec = max(recall_folds_compare_methods["XGDAG + LP"])

        sns.lineplot(x=ratios, y=recall_folds,label="NIAPU")
        if max_rec < max(recall_folds):
            max_rec = max(recall_folds)

        for METHOD in COMPARE_METHODS:
            
            if METHOD not in GUILD_METHODS:
                sns.lineplot(x=ratios, y=recall_folds_compare_methods[METHOD],label=METHOD)
            else:
                if METHOD != "NetRank":
                    sns.lineplot(x=ratios, y=recall_folds_compare_methods[METHOD],label= "GUILD-" + METHOD)
                else:
                    sns.lineplot(x=ratios, y=recall_folds_compare_methods[METHOD],label="ToppGene")

            if max_rec < max(recall_folds_compare_methods[METHOD]):
                max_rec = max(recall_folds_compare_methods[METHOD])

        plt.ylabel("Recall")
        plt.xlabel("Number of candidate genes")
        plt.legend(loc = "best")
        # plt.legend(["NIAPU P"], ["NIAPU R"])
        plt.title("Recall scores for disease " + dis_name_title)
        plt.ylim(0, max_rec + 0.01)
        
        # plt.ylim(0, max(recall_folds) + 0.1)
        plt.xticks(ratios)
        # plt.yticks(precision_folds)
        plt.savefig("Plots/recall_" + DISEASE_NAME + ".png", dpi=300)
        plt.clf()

        ######precision###
        max_prec = 0
        figure(figsize=(16, 10))
        sns.set(font_scale=1.5)
        # if XGDAG:
        #     sns.lineplot(x=ratios, y=precision_folds_compare_methods["XGDAG"],label="XGDAG")
        #     max_prec = max(precision_folds_compare_methods["XGDAG"])

        sns.lineplot(x=ratios, y=precision_folds,label="NIAPU")
        
        if max_prec < max(precision_folds):
            max_prec = max(precision_folds)

        for METHOD in COMPARE_METHODS:
           
            if METHOD not in GUILD_METHODS:
                sns.lineplot(x=ratios, y=precision_folds_compare_methods[METHOD],label=METHOD)
                
            else:
                if METHOD != "NetRank":
                    sns.lineplot(x=ratios, y=precision_folds_compare_methods[METHOD],label= "GUILD-" + METHOD)
                else:
                    sns.lineplot(x=ratios, y=precision_folds_compare_methods[METHOD],label="ToppGene")
            if max_prec < max(precision_folds_compare_methods[METHOD]):
                max_prec = max(precision_folds_compare_methods[METHOD])
        
        
        plt.ylabel("Precision")
        plt.xlabel("Number of candidate genes")
        plt.legend(loc = "best")
        plt.title("Precision scores for disease " + dis_name_title)
        # plt.legend(["NIAPU P"], ["NIAPU R"])
        #plt.ylim(0, 1)
        plt.ylim(0, max_prec + 0.01)
        plt.xticks(ratios)
        # plt.yticks(precision_folds)
        plt.savefig("Plots/precision_" + DISEASE_NAME + ".png", dpi=300)
        plt.clf()


        ######F1 score###
        max_F1 = 0
        figure(figsize=(16, 10))
        sns.set(font_scale=1.5)
        # if XGDAG:
        #     sns.lineplot(x=ratios, y=F1_folds_compare_methods["XGDAG"],label="XGDAG")
        #     max_F1 = max(F1_folds_compare_methods["XGDAG"])

        sns.lineplot(x=ratios, y=F1_folds,label="NIAPU")
        
        if max_F1 < max(F1_folds):
            max_F1 = max(F1_folds)

        for METHOD in COMPARE_METHODS:
            
            if METHOD not in GUILD_METHODS:
                sns.lineplot(x=ratios, y=F1_folds_compare_methods[METHOD],label=METHOD)
                
            else:
                if METHOD != "NetRank":
                    sns.lineplot(x=ratios, y=F1_folds_compare_methods[METHOD],label= "GUILD-" + METHOD)
                else:
                    sns.lineplot(x=ratios, y=F1_folds_compare_methods[METHOD],label="ToppGene")
            if max_F1 < max(F1_folds_compare_methods[METHOD]):
                max_F1 = max(F1_folds_compare_methods[METHOD])
        
        
        plt.ylabel("F1 score")
        plt.xlabel("Number of candidate genes")
        plt.legend(loc = "best")
        plt.title("F1 scores for disease " + dis_name_title)
        # plt.legend(["NIAPU P"], ["NIAPU R"])
        #plt.ylim(0, 1)
        plt.ylim(0, max_F1 + 0.01)
        plt.xticks(ratios)
        # plt.yticks(precision_folds)
        plt.savefig("Plots/F1_" + DISEASE_NAME + ".png", dpi=300)
        plt.clf()
        plt.close('all')