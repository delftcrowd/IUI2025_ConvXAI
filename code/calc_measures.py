import numpy as np
import os
import pandas as pd
data_folder = "anonymous_data"


def calc_mean(value_list):
	return np.mean(value_list)

def reverse_code(value, max_scale):
	return max_scale + 1 - value

# scale in [1: max_scale]
class questionnaire(object):
	
	def __init__(self, values, reverse_code_index, max_scale=6, questionnaire_size=0, value_add_one=False):
		self.questionnaire_size = questionnaire_size
		assert len(values) == self.questionnaire_size
		tp_values = []
		for i, value in enumerate(values):
			if value_add_one:
				value_ = float(value) + 1
			else:
				value_ = float(value)
			if (i + 1) in reverse_code_index:
				tp_values.append(reverse_code(value_, max_scale))
			else:
				tp_values.append(value_)
		self.values = tp_values
		self.max_scale = max_scale

	def calc_value(self):
		# x = calc_mean(self.values)
		# print("value", len(self.values), self.values, x)
		return calc_mean(self.values)


TiA_subscales = ["Reliability/Competence", "Understanding/Predictability", "Familiarity", \
				 "Intention of Developers", "Propensity to Trust", "Trust in Automation"]


class TiA_questionnaire(object):
	
	def __init__(self, values):
		self.questionnaire_size = 19
		self.max_scale = 5
		self.reverse_code_index = [5, 10, 15, 16]
		self.subscales = {
			"Reliability/Competence": [1, 6, 10, 13, 15, 19],
			"Understanding/Predictability": [2, 7, 11, 16],
			"Familiarity": [3, 17],
			"Intention of Developers": [4, 8],
			"Propensity to Trust": [5, 12, 18],
			"Trust in Automation": [9, 14]
		}
		assert len(values) == self.questionnaire_size
		tp_values = []
		for i, value in enumerate(values):
			if (i + 1) in self.reverse_code_index:
				tp_values.append(reverse_code(value, self.max_scale))
			else:
				tp_values.append(value)
		self.values = tp_values
		self.scale_dict = {}

	def calc_value(self):
		for subscale in self.subscales:
			self.scale_dict[subscale] = calc_mean([self.values[index - 1] for index in self.subscales[subscale]])
			# print(subscale, self.scale_dict[subscale])
		# print("-" * 20)
		return self.scale_dict

def calc_ATI_scale():
	filename = os.path.join(data_folder, "ATI.csv")
	df = pd.read_csv(filename)
	user_task_list = df.values.tolist()
	print("ATI len", len(user_task_list))
	user_ATI_scale = {}
	reverse_code_index = [3, 6, 8]
	max_scale = 6
	questionnaire_size = 9
	# reverse code for question 3, 6, 8
	user_ATI_time = {}
	for tuple_ in user_task_list:
		# user_id, answer_1, answer_2, answer_3, answer_4, answer_5, answer_6, answer_7, answer_8, answer_9, __ = tuple_
		# reverse code for question 3, 6, 8
		# answer_3 = 7 - answer_3
		# answer_6 = 7 - answer_6
		# answer_8 = 7 - answer_8
		# ATI_scale = (answer_1 + answer_2 + answer_3 + answer_4 + answer_5 + answer_6 + answer_7 + answer_8 + answer_9) / 9.0
		ATI_scale = questionnaire(tuple_[1:-1], reverse_code_index, max_scale=max_scale, questionnaire_size=questionnaire_size).calc_value()
		user_id = tuple_[0]
		user_ATI_scale[user_id] = ATI_scale
		user_ATI_time[user_id] = tuple_[-1]
	return user_ATI_scale, user_ATI_time



def calc_TiA_scale():
	filename = os.path.join(data_folder, "TiA_PostQ.csv")
	df = pd.read_csv(filename)
	user_task_list = df.values.tolist()
	user_TiA_scale = {}
	for tuple_ in user_task_list:
		TiA_scale = TiA_questionnaire(tuple_[1:-1]).calc_value()
		user_id = tuple_[0]
		user_TiA_scale[user_id] = TiA_scale
	return user_TiA_scale

class UserPerformance(object):
	
	def __init__(self, username):
		self.username = username
		self.performance = {}
		self.keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]

	def add_performance(self, accuracy, agreement_fraction, switching_fraction, appropriate_reliance, relative_positive_ai_reliance, relative_positive_self_reliance, group="first_group"):
		self.performance = {
			"accuracy": accuracy,
			"agreement_fraction": agreement_fraction,
			"switching_fraction": switching_fraction,
			"appropriate_reliance": appropriate_reliance,
			"relative_positive_ai_reliance": relative_positive_ai_reliance,
			"relative_positive_self_reliance": relative_positive_self_reliance
		}

	def print_information(self):
		print("-" * 17)
		print(f"User {self.username}")
		for key_ in self.keys:
			print(key_, f"{self.performance[key_]}")
		print("-" * 17)



def calc_user_reliance_measures(user, usertask_dict, advice_dict, ground_truth_dict, task_id_list):
	user_trust_list = []
	tp_correct = 0
	tp_trust = 0
	tp_agreement = 0
	initial_disagreement = 0
	tp_switch_reliance = 0
	tp_correct_initial_disagreement = 0
	positive_ai_reliance = 0
	negative_ai_reliance = 0
	positive_self_reliance = 0
	negative_self_reliance = 0
	for task_id in task_id_list:
		first_choice = usertask_dict[user][(task_id, "base")]
		second_choice = usertask_dict[user][(task_id, "advice")]
		system_advice = advice_dict[task_id]
		correct_answer = ground_truth_dict[task_id]
		# print(first_choice, second_choice, system_advice, correct_answer)
		if second_choice == system_advice:
			# agreement fraction
			tp_agreement += 1
		if first_choice != system_advice:
			# initial disagreement
			initial_disagreement += 1
			if system_advice == correct_answer:
				if second_choice == system_advice:
					# user switch to ai advice, which is correct
					positive_ai_reliance += 1
				else:
					# user don't rely on AI systems when it's correct
					negative_self_reliance += 1
			else:
				if first_choice == correct_answer:
					if second_choice == correct_answer:
						# AI system provide wrong advice, but user insist their own correct decision
						positive_self_reliance += 1
					else:
						# After wrong AI advice, users changed the decision to wrong term
						negative_ai_reliance += 1
			if second_choice == system_advice:
				tp_switch_reliance += 1
			if second_choice == correct_answer:
				tp_correct_initial_disagreement += 1
		if second_choice == correct_answer:
			tp_correct += 1
	number_of_tasks = float(len(task_id_list))
	tp_accuracy = tp_correct / number_of_tasks
	tp_agreement_fraction = tp_agreement / number_of_tasks
	if positive_ai_reliance + negative_self_reliance > 0:
		relative_positive_ai_reliance = positive_ai_reliance / float(positive_ai_reliance + negative_self_reliance)
	else:
		relative_positive_ai_reliance = 0.0

	if positive_self_reliance + negative_ai_reliance > 0:
		relative_positive_self_reliance = positive_self_reliance / float(positive_self_reliance + negative_ai_reliance)
	else:
		relative_positive_self_reliance = 0.0

	if initial_disagreement > 0:
		tp_switching_fraction= float(tp_switch_reliance) / initial_disagreement
		tp_appropriate_reliance = float(tp_correct_initial_disagreement) / initial_disagreement
	else:
		tp_switching_fraction = 0.0
		tp_appropriate_reliance = 0.0
	return tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement, relative_positive_ai_reliance, relative_positive_self_reliance
