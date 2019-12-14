import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from io import StringIO
from mlxtend.plotting import plot_decision_regions
from sklearn import linear_model
from sklearn.svm import SVR, SVC

# Config

TRAINING_FIELDS = set([
  'aimYawMaxDelta', 'aimPitchMaxDelta'
])
TRAINING_LABEL_FIELD = 'isAimbotEnabled'
CHEAT_RATE_CUTOFF = 0.5
PRINT_PROGRESS = False
PRINT_PRE_KILL_STATS = False
PRINT_FINAL_STATS = False
PRINT_ALL_DATA = False
PRINT_ANOMALY_STATS = True
PLOT_DATASETS = False
PLOT_LEARNING = False

# must be consistent with value in preprocess.py
KILL_TICK_GRANULARITY = 5

# Plotting

def plot(data, labels, clf, title):
  if not PLOT_DATASETS:
    return
  plot_decision_regions(X=data, y=labels, clf=clf, legend=2)
  plt.xlabel('Maximum Aim Yaw Change (Degrees)', size=14)
  plt.ylabel('Maximum Aim Pitch Change (Degrees)', size=14)
  plt.title(title, size=16)
  plt.show()


def plot_learning(old_stdout, my_stdout):
  if not PLOT_LEARNING:
    return
  loss_history = my_stdout.getvalue()
  loss_list = []
  for line in loss_history.split('\n'):
      if(len(line.split("loss: ")) == 1):
        continue
      loss_list.append(float(line.split("loss: ")[-1]))
  plt.figure()
  plt.plot(np.arange(len(loss_list)), loss_list)
  plt.xlabel('Epoch')
  plt.ylabel('Loss (Hinge)')
  plt.title('Loss Curve for SVM using SGD')
  plt.show()
  sys.stdout = old_stdout
  for l in loss_list:
    print(l)

# SV*

def is_number(n):
  try:
    float(n)
    return True
  except:
    return False


def parse_number(s):
  return float(s)


def extract_features(d):
  return [float(v) for k, v in d.items() if k in TRAINING_FIELDS and is_number(v)]


def extract_label(d, should_bias=True):
  label_field = d[TRAINING_LABEL_FIELD]

  # bias
  if should_bias and (float(d['aimYawMaxDelta']) > 40 or float(d['aimPitchMaxDelta']) > 10):
    return 1

  if not is_number(label_field):
    return 0
  return int(label_field)


def create_kill_test_data(kill_tick_data, should_bias=True):
  test_data = []
  test_labels = []
  for d in kill_tick_data:
    test_data.append(extract_features(d))
    test_labels.append(extract_label(d, should_bias))
  return np.array(test_data), np.array(test_labels)


def score_group(predictions, labels):
  count = 0
  true_positives = 0
  false_negatives = 0
  false_positives = 0
  for i, prediction in enumerate(predictions):
    label = labels[i]
    if prediction == 1:
      count += 1
    if label == prediction:
      true_positives += 1
    elif label == 0 and prediction == 1:
      false_negatives += 1
    elif label == 1 and prediction == 0:
      false_positives += 1
  accuracy = true_positives / len(predictions) if true_positives > 0 else 0
  precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0
  recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0
  cheat_rate =  count / len(predictions)
  f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
  return accuracy, precision, recall, f1, cheat_rate


def main():
  # read output data from preprocess.py
  kill_ticks_frame = pd.read_csv('data/output.csv')
  kill_ticks_list = list(kill_ticks_frame.T.to_dict().values())

  # project output data to fields we care about
  training_data = []
  training_labels = []
  for kill_tick_data in kill_ticks_list:
    features = extract_features(kill_tick_data)
    if not features:
      continue
    training_data.append(features)
    training_labels.append(extract_label(kill_tick_data))
  
  n_samples = len(training_data)
  n_features = len(training_data[0])
  if n_samples == 0 or n_features == 0:
    print('No samples or features... exiting')
    return

  training_data = np.array(training_data)
  training_labels = np.array(training_labels)

  if PRINT_PROGRESS:
    print(f'Training with {n_features} features and {n_samples} samples.')
  # fit training data for SV*
  # sv = SVC(gamma='auto', kernel='poly')
  # sv = SVR(gamma='scale', C=1.0, epsilon=0.2)

  my_stdout = StringIO()
  if PLOT_LEARNING:
    old_stdout = sys.stdout
    sys.stdout = my_stdout

  sv = linear_model.SGDClassifier(verbose=1 if PLOT_LEARNING else 0)
  clf = sv.fit(training_data, training_labels)
  if PRINT_PROGRESS:
    print('Trained model.')

  # Plot learning
  if PLOT_LEARNING:
    plot_learning(old_stdout, my_stdout)

  # Plot all training data
  plot(training_data, training_labels, clf, 'SVM Decision Region Boundary Using SGD')

  # read test data from preprocess.py
  test_kill_ticks_frame = pd.read_csv('data/output_test.csv')
  test_kill_ticks_list = list(test_kill_ticks_frame.T.to_dict().values())
  if len(test_kill_ticks_list) == 0:
    print('No test kill tick data... exiting')
    return
  test_data, test_labels = create_kill_test_data(test_kill_ticks_list)

  if PRINT_ALL_DATA or PRINT_ANOMALY_STATS:
    all_data = training_data.tolist() + training_data.tolist()
    unique_points = set([(point[0], point[1]) for point in all_data])
    
    # print stats in format for CSV for R
    if PRINT_ALL_DATA:
      print('aimYawMaxDelta,aimPitchMaxDelta')
      for point in all_data:
        print(f'{point[0]},{point[1]}')
   
     # determine anomaly stats
    if PRINT_ANOMALY_STATS:
      unique_points_arr = [[p[0], p[1]] for p in unique_points]
      outliers = set([5,9,19,42,47,48,65,66,67,76,77,88,89,96,97,99,100,103])
      all_data_labels = [1 if i in outliers else 0 for i in range(len(unique_points_arr))]
      d_accuracy, d_precision, d_recall, d_f1, d_cheat_rate = score_group(sv.predict(unique_points_arr), all_data_labels)
      print('===============')
      print('Anomaly Results')
      print('===============')
      print(f'Number of ticks: {len(unique_points)}')
      print(f'Accuracy: {d_accuracy}')
      print(f'Precision: {d_precision}')
      print(f'Recall: {d_recall}')
      print(f'F1: {d_f1}')
      print(f'Score: {clf.score(unique_points_arr, all_data_labels)}')
      print(f'Cheat Rate: {d_cheat_rate}')
      print('=============')
      print()

  # Plot all test data
  plot(test_data, test_labels, clf, 'Test Set on Decision Region Boundaries')

  # calculate performance of SV* prediction
  predictions = sv.predict(test_data)
  accuracy, precision, recall, f1, cheat_rate = score_group(predictions, test_labels)
  if PRINT_PRE_KILL_STATS:
    print('=================')
    print('Aggregate Results')
    print('=================')
    print(f'Number of ticks: {len(predictions)}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'Score: {clf.score(test_data, test_labels)}')
    print(f'Cheat Rate: {cheat_rate}')
    print()

  # calculate performance for individual kills (groups of ticks)
  # assumes that preprocess.py separated kills by players
  groups = []
  current_group = []
  for test_kill_tick_data in test_kill_ticks_list:
    if len(current_group) == 0:
      current_group.append(test_kill_tick_data)
    elif abs(test_kill_tick_data['tick'] - current_group[-1]['tick']) == KILL_TICK_GRANULARITY:
      current_group.append(test_kill_tick_data)
    else:
      groups.append(current_group)
      current_group = [test_kill_tick_data]
  groups.append(current_group)


  k_cheat_count = 0
  k_true_positives = 0
  k_false_negatives = 0
  k_false_positives = 0
  for i, g_tick_data in enumerate(groups):
    if len(g_tick_data) == 0:
      print(f'No tick data for kill {i}, skipping...')
      continue
    g_test_data, g_test_labels = create_kill_test_data(g_tick_data)
    g_accuracy, g_precision, g_recall, g_f1, g_cheat_rate = score_group(sv.predict(g_test_data), g_test_labels)
    g_confidences = sv.decision_function(g_test_data)
    
    # compute statistics for this prediction
    was_cheating = len([d for d in g_tick_data if d[TRAINING_LABEL_FIELD]]) > 0
    predicted_cheating = g_cheat_rate >= CHEAT_RATE_CUTOFF
    if predicted_cheating:
      k_cheat_count += 1
    if was_cheating == predicted_cheating:
      k_true_positives += 1
    elif not was_cheating and predicted_cheating:
      k_false_positives += 1
    elif was_cheating and not predicted_cheating:
      k_false_negatives += 1

    if PRINT_PRE_KILL_STATS:
      print('=============')
      print(f'Player {g_tick_data[0]["playerId"]} kill')
      print('=============')
      print(f'Number of ticks: {len(g_tick_data)}')
      print(f'Cheating: {"Yes" if was_cheating else "No"}')
      print(f'Accuracy: {g_accuracy}')
      print(f'Precision: {g_precision}')
      print(f'Recall: {g_recall}')
      print(f'F1: {g_f1}')
      print(f'Score: {clf.score(g_test_data, g_test_labels)}')
      print(f'Cheat Rate: {g_cheat_rate}')
      print(f'Average Confidence: {sum(g_confidences) / len(g_confidences)}')
      print('=============')
      print()

  # treat each group as a single datapoint that's been classified now and report statistics
  k_accuracy = k_true_positives / len(groups) if len(groups) > 0 else 0
  k_precision = k_true_positives / (k_true_positives + k_false_positives) if k_true_positives > 0 else 0
  k_recall = k_true_positives / (k_true_positives + k_false_negatives) if k_true_positives > 0 else 0
  k_cheat_rate = k_cheat_count / len(groups)
  k_f1 = 2 * ((k_precision * k_recall) / (k_precision + k_recall)) if k_precision + k_recall > 0 else 0
  if PRINT_FINAL_STATS:
    print('==================')
    print('Kill-Level Results')
    print('==================')
    print(f'Number of kills: {len(groups)}')
    print(f'Accuracy: {k_accuracy}')
    print(f'Precision: {k_precision}')
    print(f'Recall: {k_recall}')
    print(f'F1: {k_f1}')
    print(f'Cheat Rate: {k_cheat_rate}')
    print()


if __name__ == '__main__':
  main()
