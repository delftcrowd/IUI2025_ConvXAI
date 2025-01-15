import json
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import csv
from calc_measures import questionnaire, calc_user_reliance_measures

def collect_file_users(filename):
    # df = pd.read_csv(filename, usecols=['user_id'])
    df = pd.read_csv(filename, usecols=['userid'], quoting=csv.QUOTE_MINIMAL)
    user_list = df.values.tolist()
    user_set = set([item[0] for item in user_list])
    # f = open(filename)
    # f.readline()
    # user_set = set()
    # num_lines = 0
    # for line in f:
    # 	prolific_id = line.strip().split(",")[0]
    # 	user_set.add(prolific_id)
    # 	num_lines += 1
    print(filename, len(user_set))
    return user_set


def load_task_info(filename="../task_details.csv"):
    f = open(filename)
    f.readline()
    advice_dict = {}
    ground_truth_dict = {}
    for line in f:
        __, task_id, xx, model_prediction, ground_truth = line.strip().split(",")
        advice_dict[task_id] = model_prediction.lower()
        ground_truth_dict[task_id] = ground_truth.lower()
    return advice_dict, ground_truth_dict

page_order = ['consent', 'ati', 'atiExplanationTest', 'attn', 'survey1', 'survey2', 'survey3', 'survey4']
# consent: mlbackground
# ati: ati1 - ati9, attnAti, correct answer = largely agree (4)
# atiExplanationTest: atiExplainerQuestion, correct answer = decision tree (5)
# attn: attnBeforeExplainDecision, correct answer = true
# ---------------
# All surveys in 5-point Likert
# survey 1: q1, q2, q3, q4a, q4b, q5
# q1: perceived feature understanding
# q2 + 14b: Explanation Completeness
# q3: consistent / coherence? to further check
# To what extent were the explanations you received consistent with your initial expectations?
# q4a: understanding of the AI system
# q5: Clarity
# ---------------
# survey 2: q6 - q11
# q6: learning effect, the first question on 
# My understanding of AI system and decision criteria improve over the tasks.
# other 6 questions, q7, q8a, q8b, q9, q10, q11 - Reliability / Completeness
# invert q8b, q10
# ---------------
# survey 3: q12 - q22, attentionSurvey, correct answer = largely disagree (1)
# q12 - q15,  TiA- understandability / predictability
# invert q13, q15
# q16 - q18, TiA-Propensity to trust
# invert q16
# q19, q20 - TiA-Trust in Automation
# q21, q22 - TiA-Familiarity
# ---------------
# survey 4: q23 - q34
# user engagement questionnaire
# q26, q27, q28
# ---------------
task_template = ['taskx_pre', 'taskx_post']

background_map = {
    "true": 1,
    "false": 0
}

shap_ranking = {
    "task1" : ['Credit History', 'Loan Amount', 'Applicant Income', 'Coapplicant Income', 'Property Area', 'Married', 'Dependents', 'Loan Amount Term', 'Self Employed', 'Gender', 'Education'],
    "task2" : ['Credit History', 'Loan Amount', 'Applicant Income', 'Property Area', 'Married', 'Dependents', 'Loan Amount Term', 'Education', 'Gender', 'Self Employed', 'Coapplicant Income'],
    "task3" : ['Loan Amount', 'Applicant Income', 'Credit History', 'Married', 'Coapplicant Income', 'Self Employed', 'Property Area', 'Education', 'Gender', 'Dependents', 'Loan Amount Term'],
    "task4" : ['Credit History', 'Dependents', 'Coapplicant Income', 'Applicant Income', 'Property Area', 'Loan Amount Term', 'Education', 'Loan Amount', 'Married', 'Self Employed', 'Gender'],
    "task5" : ['Credit History', 'Applicant Income', 'Loan Amount', 'Coapplicant Income', 'Loan Amount Term', 'Self Employed', 'Dependents', 'Property Area', 'Married', 'Education', 'Gender'],
    "task6" : ['Loan Amount', 'Property Area', 'Credit History', 'Married', 'Gender', 'Dependents', 'Applicant Income', 'Self Employed', 'Education', 'Coapplicant Income', 'Loan Amount Term'],
    "task7" : ['Credit History', 'Loan Amount', 'Loan Amount Term', 'Coapplicant Income', 'Applicant Income', 'Self Employed', 'Married', 'Education', 'Property Area', 'Gender', 'Dependents'],
    "task8" : ['Loan Amount', 'Credit History', 'Property Area', 'Applicant Income', 'Education', 'Coapplicant Income', 'Self Employed', 'Loan Amount Term', 'Dependents', 'Gender', 'Married'],
    "task9" : ['Credit History', 'Applicant Income', 'Coapplicant Income', 'Property Area', 'Dependents', 'Married', 'Education', 'Loan Amount', 'Self Employed', 'Gender', 'Loan Amount Term'],
    "task10" : ['Credit History', 'Loan Amount', 'Dependents', 'Applicant Income', 'Coapplicant Income', 'Property Area', 'Education', 'Self Employed', 'Loan Amount Term', 'Married', 'Gender']
}

def ndcg_compute(reference, user_choice):
    relevance = {}
    for j in range(len(reference)):
        relevance[reference[j]] = len(reference) - j
    idcg = 11/np.log2(1+1) + 10/np.log2(2+1) + 9/np.log2(3+1)
    dcg = 0
    for i in range(len(user_choice)):
        dcg = dcg + relevance[user_choice[i]]/np.log2(i+1+1)
    return dcg/idcg

def load_user_data(filename="pilot_conversationalXAI.csv", reserved_users=None):
    used_cols = ['userid', 'group', 'page', 'data']
    df = pd.read_csv(filename, usecols=used_cols)
    user_data_list = df.values.tolist()
    user_set = set()
    user2condition = {}
    user_ATI_scale = {}
    user_TiA_scale = {}
    user_engagement_scale = {}
    condition_users = {}
    user_task_order = {}
    user_task_dict = {}
    user_confidence_dict = {}
    user_time_dict = {}
    user_feature_dict = {}
    user_userfulness_explanation_dict = {}
    user_ml_background = {}
    user_attention_check_wrong = {}
    user_perceived_feature_understanding = {}
    user_objective_feature_understanding = {}
    user_coherence = {}
    user_clarity = {}
    user_xai_usage = {}
    user_completeness = {}
    user_understanding_of_AIsys = {}
    user_learning_effect = {}
    conditions = ["control", "dashboard", "chatxai", "chatxaiboost", "chatxaiAuto"]
    for condition in conditions:
        condition_users[condition] = set()
	
    def load_survey1(user_id, data_dict):
        perceived_feature_understanding = int(data_dict["q1"])
        completeness = (int(data_dict["q2"]) + int(data_dict["q4b"])) / 2.0
        coherence = int(data_dict["q3"])
        try:
            understanding_of_system = int(data_dict["q4a"])
        except:
            print(data_dict)

        user_understanding_of_AIsys[user_id] = understanding_of_system
        clarity = int(data_dict["q5"])
        # add user_dict
        user_perceived_feature_understanding[user_id] = perceived_feature_understanding
        user_completeness[user_id] = completeness
        user_coherence[user_id] = coherence
        user_clarity[user_id] = clarity

    def load_survey2(user_id, data_dict):
        learning_effect = int(data_dict["q6"])
        user_learning_effect[user_id] = learning_effect
        tp_keyps = ["q7", "q8a", "q8b", "q9", "q10", "q11"]
        reverse_code_index = [3, 5]
        max_scale = 5
        questionnaire_size = 6
        data_list = []
        for tp_key in tp_keyps:
            if tp_key == "q8a" or tp_key == "q8b":
                if tp_key not in data_dict:
                    print(data_dict)
                    continue
            data_list.append(data_dict[tp_key])
        user_TiA_scale[user_id]["Reliability/Competence"] = questionnaire(data_list, reverse_code_index, max_scale=max_scale, 
            questionnaire_size=len(data_list), value_add_one=False).calc_value()

    def load_survey3(user_id, data_dict):
        tp_keys = ["q12", "q13", "q14", "q15"]
        reverse_code_index = [2, 4]
        max_scale = 5
        questionnaire_size = 4
        data_list = []
        for tp_key in tp_keys:
            data_list.append(data_dict[tp_key])
        user_TiA_scale[user_id]["Understandability/Predictability"] = questionnaire(data_list, reverse_code_index, max_scale=max_scale, 
            questionnaire_size=len(data_list), value_add_one=False).calc_value()
        tp_keys = ["q16", "q17", "q18"]
        reverse_code_index = [1]
        data_list = []
        for tp_key in tp_keys:
            data_list.append(data_dict[tp_key])
        user_TiA_scale[user_id]["Propensity to Trust"] = questionnaire(data_list, reverse_code_index, max_scale=max_scale, 
            questionnaire_size=len(data_list), value_add_one=False).calc_value()
        # print(type(data_dict["q19"]), data_dict["q19"])
        user_TiA_scale[user_id]["Trust in Automation"] = (int(data_dict["q19"]) + int(data_dict["q20"])) / 2.0
        user_TiA_scale[user_id]["Familiarity"] = (int(data_dict["q21"]) + int(data_dict["q22"])) / 2.0


    def load_ati_data(user_id, data_dict):
        reverse_code_index = [3, 6, 8]
        max_scale = 6
        questionnaire_size = 9
        data_list = []
        for i in range(9):
            tp_key = "ati%d"%(i+1)
            data_list.append(data_dict[tp_key])
        ATI_scale = questionnaire(data_list, reverse_code_index, max_scale=max_scale, 
            questionnaire_size=questionnaire_size, value_add_one=False).calc_value()
        user_ATI_scale[user_id] = ATI_scale

    def load_user_engagement(user_id, data_dict):
        reverse_code_index = [4, 5, 6]
        max_scale = 5
        questionnaire_size = 12
        data_list = []
        for i in range(23, 35):
            # q23 - q35
            tp_key = "q%d"%i
            data_list.append(data_dict[tp_key])
        # overall / average user engagement
        user_engagement = questionnaire(data_list, reverse_code_index, max_scale=max_scale, 
            questionnaire_size=questionnaire_size, value_add_one=False).calc_value()
        # if user_id == '6014c1b18c19b3524a3ef183':
        # 	print("engagement", user_id, user_engagement)
        user_engagement_scale[user_id] = user_engagement
        
    def analyze_chatlog_new(data_dict):
        api_calls = data_dict["explainerData"]["api_calls"]
        chat_history = data_dict["explainerData"]["chat_history"]
        user_queries = []
        action_sequence = []
        for message in chat_history:
            if message.startswith("[user] "):
                # user message
                user_message = message[7:]
                user_queries.append(user_message)
            else:
                # agent message, message.startswith("[agent] ")
                agent_message = message[8:]
        explainer_counter = {}
        explainers = ["pdp", "shap", "whatIf", "decisionTree", "counterFactual"]
        for explainer in explainers:
            explainer_counter[explainer] = 0
        function_call_history = data_dict["explainerData"]["function_call_history"]
        for function_call in function_call_history:
            function_name = function_call["name"]
            # ask_shap, ask_what_if, ask_pdp, ask_mace, ask_decision_tree, ask_prediction
            # as what_if and ask_prediction work similarly generating new prediction given an input, we take both as what if explainer
            if function_name not in ["ask_shap", "ask_what_if", "ask_pdp", "ask_mace", "ask_decision_tree", "ask_prediction"]:
                print("LLM Agent condition, Unknown action used", function_name)
                action = "unknown"
            if function_name == "ask_shap":
                action = "shap"
            if function_name == "ask_pdp":
                action = "pdp"
                # for pdp, function_call["arguments"]["feature"] point out feature used
            if function_name == "ask_what_if" or function_name == "ask_prediction":
                action = "whatIf"
            if function_name == "ask_decision_tree":
                action = "decisionTree"
            if function_name == "ask_mace":
                action = "counterFactual"
            explainer_counter[action] += 1
            action_sequence.append(action)
        # count the number of explainers used
        response = {
            "api_calls": api_calls,
            "action_sequence": action_sequence,
            "explainer_counter": explainer_counter,
            "user_queries": user_queries
        }
        return response

    def analyze_chatlog(data_dict):
        chat_journey = data_dict["explainerData"]["chatJourney"]
        number_operations = len(chat_journey)
        # initial_step = int(chat_journey[0]["step"])
        action_sequence = []
        explainer_counter = {}
        explainers = ["pdp", "shap", "whatIf", "decisionTree", "counterFactual"]
        for explainer in explainers:
            explainer_counter[explainer] = 0
        for index, tp_op in enumerate(chat_journey):
            # tp_step = tp_op["step"] - initial_step + 1
            # print(tp_op)
            tp_level = tp_op["level"]
            if tp_level == "predLv":
                # it's the beginning of each conversation
                continue
            # action = "unknown"
            if tp_level not in ["globalExplainerResultLv", "counterfactualLv", "featureImpLv", "whatIfResultLv"]:
                print("Unknown level", tp_level)
                # print("LLM Agent condition, Unknown action used", tp_level)
                action = "unknown"
                assert False
            else:
                if tp_level == "globalExplainerResultLv":
                    assert tp_op["payload"]["data"]["instances"][0]["operation"] == "pdp"
                    # explainer_counter["pdp"] += 1
                    action = "pdp"
                if tp_level == "counterfactualLv":
                    assert tp_op["payload"]["data"]["instances"][0]["operation"] == "mace"
                    # explainer_counter["counterFactual"] += 1
                    action = "counterFactual"
                if tp_level == "featureImpLv":
                    assert tp_op["payload"]["data"]["instances"][0]["operation"] == "shap"
                    # explainer_counter["shap"] += 1
                    action = "shap"
                if tp_level == "whatIfResultLv":
                    assert tp_op["payload"]["data"]["instances"][0]["operation"] == "whatif"
                    # explainer_counter["whatIf"] += 1
                    action = "whatIf"
            explainer_counter[action] += 1
            action_sequence.append(action)
        metadata = data_dict["explainerData"]["chatSessionMetadata"]
        # decision tree is only recorded in the metadata
        explainer_counter["decisionTree"] = metadata["decisionTree"]
        for explainer in explainers:
            try:
                assert explainer_counter[explainer] == metadata[explainer]
            except:
                # when the counter of action sequence differ from metadata, use metadata
                explainer_counter[explainer] = metadata[explainer]
                # print(explainer)
                # print(explainer_counter[explainer], type(explainer_counter[explainer]))
                # print(metadata[explainer], type(metadata[explainer]))
                # print(explainer_counter, metadata)
                # print(explainer_counter[explainer], metadata[explainer])
        # 		# sys.exit(-1)
        # 		# To check with Nilay, there is some issue; decisionTree is never observed in the steps
        # 		# WhatIf sometimes is also not observed
        response = {
            "action_sequence": action_sequence,
            "explainer_counter": explainer_counter
        }
        return response

    def load_task_data(user_id, group, page, data_dict):
        task_id, phase = page.split("_")
        if phase == "pre":
            answer_type = "base"
            user_choice = data_dict["beforeExplainDecision"]
            features = data_dict["beforeExplainFeatures"]
            confidence = data_dict["beforeExplainConfidence"]
        if phase == "post":
            answer_type = "advice"
            user_choice = data_dict["postExplainDecision"]
            features = data_dict["postExplainFeatures"]
            confidence = data_dict["postExplainConfidence"]
            user_task_order[user_id].append(task_id)
            if group != "control":
                # for conditions with explanation
                usefulness_explanation = data_dict["postExplainReliability"]
                user_userfulness_explanation_dict[user_id][task_id] = int(usefulness_explanation)
            if group in ["chatxai", "chatxaiboost"]:
                # a dict
                user_xai_usage[user_id][task_id] = analyze_chatlog(data_dict)
                # user_xai_usage[user_id][task_id] = {}
            if group == "chatxaiAuto":
                user_xai_usage[user_id][task_id] = analyze_chatlog_new(data_dict)
        user_objective_feature_understanding[user_id][task_id] = ndcg_compute(reference=shap_ranking[task_id], user_choice=features)
        user_task_dict[user_id][(task_id, answer_type)] = user_choice
        user_confidence_dict[user_id][(task_id, answer_type)] = int(confidence)
        user_feature_dict[user_id][(task_id, answer_type)] = features

    for user_id, group, page, data in user_data_list:
        if user_id == "0":
            continue
        data_dict = json.loads(data)
        # print(user_id, group, page, data)
        if user_id not in user2condition:
            # The first time that one user_id appears
            # initialize relevant variables:
            user2condition[user_id] = group
            condition_users[group].add(user_id)
            user_attention_check_wrong[user_id] = 0
            if user_id not in user_task_dict:
                user_ATI_scale[user_id] =		{}
                user_TiA_scale[user_id] = 		{}
                user_task_dict[user_id] 		= {}
                user_objective_feature_understanding[user_id] = {}
                user_confidence_dict[user_id] 	= {}
                user_task_order[user_id]		= []
                user_time_dict[user_id] 		= {}
                user_feature_dict[user_id] 		= {}
            if group in ["chatxai", "chatxaiboost", "chatxaiAuto"]:
                user_xai_usage[user_id]			= {}
            if group != "control":
                user_userfulness_explanation_dict[user_id] = {}
        # if page.startswith("training_"):
        # 	skip pages used to train users with XAI methods
        if page == "ati":
            # the first attention check, in pre-task questionnaire
            if "attnAti" not in data_dict:
                print(user_id, "no ATI attention check")
                # print(data_dict)
            else:
                try:
                    tp_answer = int(data_dict["attnAti"])
                    assert isinstance(tp_answer, int)
                    # Largely agree
                    if tp_answer != 5:
                        user_attention_check_wrong[user_id] += 1
                except:
                    print(data_dict, "Error")
            load_ati_data(user_id, data_dict)
        if page == "atiExplanationTest":
            # the second attention check, to check user understanding of XAI methods
            # participants in control condition will not have it
            tp_answer = int(data_dict["atiExplainerQuestion"])
            assert isinstance(tp_answer, int)
            if tp_answer != 5:
                # decision tree
                user_attention_check_wrong[user_id] += 1
        if page == "attn":
            # attention check in the tasks
            if "attnBeforeExplainDecision" not in data_dict:
                print(user_id, "no task attention check")
            else:
                tp_answer = data_dict["attnBeforeExplainDecision"]
                assert tp_answer in ["true", "false"]
                if tp_answer == "false":
                    # user should choose true
                    user_attention_check_wrong[user_id] += 1
        if page == "consent":
            user_ml_background[user_id] = background_map[data_dict["mlbackground"]]
            # Yes, No
        if page.startswith("task"):
            load_task_data(user_id, group, page, data_dict)
        if page == "survey1":
            load_survey1(user_id, data_dict)
        if page == "survey2":
            load_survey2(user_id, data_dict)
        if page == "survey3":
            # The last attention check
            tp_answer = int(data_dict["attentionSurvey"])
            assert isinstance(tp_answer, int)
            # strongly disagree
            if tp_answer != 1:
                user_attention_check_wrong[user_id] += 1
            load_survey3(user_id, data_dict)
        if page == "survey4":
            load_user_engagement(user_id, data_dict)
            # print(user_id, user_ATI_scale[user_id])

    def check_valid_users():
        valid_users = set()
        # print(user2condition)
        for user in user2condition:
            flag = False
            if user not in user_ml_background:
                # print(f"{user} missing ml background")
                continue
            if user not in user_ATI_scale:
                # print(f"{user} missing ATI scale")
                continue
            TiA_subscales = ["Reliability/Competence", "Understandability/Predictability", 
            "Propensity to Trust", "Trust in Automation", "Familiarity"]
            for subscale in TiA_subscales:
                if subscale not in user_TiA_scale[user]:
                    # print(f"{user} missing TiA subscale {subscale}")
                    flag = True
                    break
            if user_attention_check_wrong[user] > 0:
                # print(f"{user} failed {user_attention_check_wrong[user]} attention checks")
                flag = True
            if len(user_task_dict[user]) < 20:
                # print(f"{user} with incomplete task records")
                flag = True
            if user2condition[user] != "control":
                if user not in user_perceived_feature_understanding:
                    # print(f"{user} missing perceived_feature_understanding")
                    flag = True
                if user not in user_completeness:
                    # print(f"{user} missing user_completeness")
                    flag = True
                if user not in user_coherence:
                    # print(f"{user} missing user_coherence")
                    flag = True
                if user not in user_clarity:
                    # print(f"{user} missing user_clarity")
                    flag = True
                if user not in user_understanding_of_AIsys:
                    # print(f"{user} missing understanding_of_AIsys")
                    flag = True
            if user not in user_engagement_scale:
                # print(f"{user} missing engagement scale")
                flag = True
            if flag:
                continue
            valid_users.add(user)
        return valid_users


    valid_users = check_valid_users()
    advice_dict, ground_truth_dict = load_task_info()
    print("{} valid participants".format(len(valid_users)))
    user_count = {}
    for condition in conditions:
        user_count[condition] = 0
    for user in valid_users:
        user_count[user2condition[user]] += 1
    print(user_count)
    task_id_list = [f"task{x}" for x in range(1, 11)]
    # print(task_id_list)
    reliance_measures = {}
    blind_trust = 0
    blind_trust_count = {}
    for condition in conditions:
        blind_trust_count[condition] = 0
    tp_data = {}
    for user in valid_users:
        tp_correct, tp_agreement_fraction, tp_switch_fraction, tp_appropriate_reliance, initial_disagreement, \
        relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, user_task_dict, advice_dict, ground_truth_dict, task_id_list)
        tp_accuracy = tp_correct / 10.0
        reliance_measures[user] = {}
        reliance_measures[user]["accuracy"] = tp_accuracy
        reliance_measures[user]["agreement_fraction"] = tp_agreement_fraction
        reliance_measures[user]["switch_fraction"] = tp_switch_fraction
        reliance_measures[user]["accuracy-wid"] = tp_appropriate_reliance
        reliance_measures[user]["RAIR"] = relative_positive_ai_reliance
        reliance_measures[user]["RSR"] = relative_positive_self_reliance
        # for user in reliance_measures:
        # print(user)
        if tp_accuracy == 0.7 and tp_agreement_fraction == 1.0:
            blind_trust += 1
            blind_trust_count[user2condition[user]] += 1
        # print("Accuracy: {:.1f}, AF: {:.3f}, SF: {:.3f}, Accuracy-wid: {:.2f}, RAIR: {:.3f}, RSR: {:.3f}".format(tp_accuracy, \
        # 	tp_agreement_fraction, tp_switch_fraction, tp_appropriate_reliance, relative_positive_ai_reliance, relative_positive_self_reliance))
        # print("ATI: {:.2f}, TiA-Trust: {:.2f}, TiA-Propensity: {:.2f}".format(user_ATI_scale[user], user_TiA_scale[user]["Trust in Automation"], user_TiA_scale[user]["Propensity to Trust"]))
        # print("TiA-Familiarity: {:.2f}, TiA-U/P: {:.2f}, TiA-R/C: {:.2f}".format(user_TiA_scale[user]["Familiarity"], user_TiA_scale[user]["Understandability/Predictability"], user_TiA_scale[user]["Reliability/Competence"]))
        # print("User engagement: {:.3f}".format(user_engagement_scale[user]))
        # print("User completeness: {:.3f}".format(user_completeness[user]))
        # print("User coherence: {:.3f}".format(user_coherence[user]))
        # print("User clarity: {:.3f}".format(user_clarity[user]))
        # print(user_confidence_dict[user])
        try:
            tp_data[user] = {
                "condition": user2condition[user],
                "reliance_measures": reliance_measures[user],
                "Trust_in_automation": user_TiA_scale[user],
                "user_engagement_scale": user_engagement_scale[user],
                "mlbackground": user_ml_background[user],
                "ATI": user_ATI_scale[user],
                "confidence": user_confidence_dict[user],
                "task_order": user_task_order[user]
            }
        except:
            print("{} in user2condition? {}".format(user, user in user2condition))
            print("{} in reliance_measures? {}".format(user, user in reliance_measures))
            print("{} in Trust_in_automation? {}".format(user, user in user_TiA_scale))
            print("{} in user_engagement_scale? {}".format(user, user in user2condition))
            print("{} in ATI? {}".format(user, user in user_ATI_scale))
            print("{} in confidence? {}".format(user, user in user_confidence_dict))
            print("{} in task_order? {}".format(user, user in user_task_order))
            sys.exit(-1)
        if user2condition[user] in ["chatxai", "chatxaiboost", "chatxaiAuto"]:
            # print(user, user2condition[user])
            if user not in user_xai_usage:
                print("Error here, with {} in group {}".format(user, user2condition[user]))
                assert False
            else:
                tp_data[user]["xai_usage"] = user_xai_usage[user]
        if user2condition[user] != "control":
            usefulness_list = []
            objective_feature_understanding_list = []
            for task_id in user_userfulness_explanation_dict[user]:
                usefulness_list.append(user_userfulness_explanation_dict[user][task_id])
                objective_feature_understanding_list.append(user_objective_feature_understanding[user][task_id])
            # print(usefulness_list)
            avg_usefulness = np.mean(usefulness_list)
            avg_objective_feature_understanding = np.mean(objective_feature_understanding_list)
            # user_feature_dict[user_id][(task_id, answer_type)] = features
            feature_switch_list = []
            for task_id in user_task_order[user]:
                features_initial = user_feature_dict[user][(task_id, "base")]
                features_final = user_feature_dict[user][(task_id, "advice")]
                set1 = set(features_initial)
                set2 = set(features_final)
                feature_switch = 3 - len(set1 & set2)
                # Feature switch = 3 - number of features overlapped
                feature_switch_list.append(feature_switch)
            avg_feature_switch = np.mean(feature_switch_list)
            tp_data[user]["explanation_understanding"] = {
                    "perceived_feature_understanding": user_perceived_feature_understanding[user],
                    "completeness": user_completeness[user],
                    "coherence": user_coherence[user],
                    "clarity": user_clarity[user],
                    "learning_effect": user_learning_effect[user],
                    "understanding_of_system": user_understanding_of_AIsys[user],
                    "usefulness_explanation": avg_usefulness,
                    "objective_feature_understanding": avg_objective_feature_understanding,
                    "feature_switch": avg_feature_switch
            }
    print("{} participants blindly rely on AI advice".format(blind_trust))
    print(blind_trust_count)
    # import json
    # print(len(tp_data))
    # with open("../original_data/measures.json", "w") as f:
    # 	json.dump(tp_data, f, indent=4)
    return valid_users, tp_data, user_task_dict

if __name__ == "__main__":
    # advice_dict, ground_truth_dict = load_task_info()
    load_user_data(filename="../original_data/xailabdata_all.csv")













