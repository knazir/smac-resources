import argparse
import collections
import csv
import os
import pandas as pd
import sys

# Arguments
parser = argparse.ArgumentParser(description='Preprocess collected data')
parser.add_argument('--test', help='whether to look for (ticks|kills)_test.csv files', action='store_true')
parser.add_argument('--games', help='the number of games played, each having a separate tick and kill file')
args = parser.parse_args()

# Config

TEST = args.test
NUM_GAMES = int(args.games or 1)

TICK_RATE = 60
KILL_WINDOW_PRIOR_SECONDS = 3
KILL_WINDOW_POST_SECONDS = 0
KILL_PRIOR_NUM_TICKS = int(KILL_WINDOW_PRIOR_SECONDS * TICK_RATE)
KILL_POST_NUM_TICKS = int(KILL_WINDOW_POST_SECONDS * TICK_RATE)
GAME_TICK_OFFSET = 1000

# must be a factor of TICK_RATE * number of ticks to record
KILL_TICK_GRANULARITY = 5

# Constants

NON_DELTA_FIELDS = set([
  # bookkeeping
  'tick', 'teamIndex', 'score', 'playerId',

  # shouldn't be deltas
  'inAirTime', 'jumpTime',

  # booleans
  'isAimbotEnabled', 'isRapidFireEnabled', 'buttonJump', 'buttonBoost',
  'buttonPrimaryFire', 'buttonSecondaryFire', 'buttonReload', 'buttonMelee',
  'buttonUse', 'buttonAbility1', 'buttonAbility2', 'buttonAbility3'
])

def is_number(n):
  try:
    float(n)
    return True
  except:
    return False

def mod(x, m):
  if m < 0:
    m = -m
  r = x % m
  return r + m if r < 0 else r

def circularDiff(a, b):
  return min(mod(a - b ,360), mod(b - a, 360))

def absDiff(a, b):
  return abs(b - a)

CUSTOM_DIFFS = {
  'rotation': circularDiff,
  'aimYaw': circularDiff,
  'aimPitch': absDiff,
  'moveYaw': absDiff
}

# Preprocessing

def ticks_filename(game_num):
  prefix = f'ticks_{game_num}'
  suffix = '_test.csv' if TEST else '.csv'
  return f'data/{prefix}{suffix}'


def kills_filename(game_num):
  prefix = f'kills_{game_num}'
  suffix = '_test.csv' if TEST else '.csv'
  return f'data/{prefix}{suffix}'


def update_ticks(items, last_tick):
  tick_offset = last_tick + GAME_TICK_OFFSET
  for item in items:
    item['tick'] += tick_offset


def output_filename():
  return 'data/output_test.csv' if TEST else 'data/output.csv'


def main():
  if os.path.exists(output_filename()):
    os.remove(output_filename())
  last_tick = -GAME_TICK_OFFSET
  for game_num in range(1, NUM_GAMES + 1):
    # setup tick data
    ticks_frame = pd.read_csv(ticks_filename(game_num), error_bad_lines=False)
    tick_list = list(ticks_frame.T.to_dict().values())
    update_ticks(tick_list, last_tick)
    ticks_map = collections.defaultdict(dict)
    for tick_data in tick_list:
      ticks_map[tick_data['tick']][tick_data['playerId']] = tick_data

    # setup kill data
    kills_frame = pd.read_csv(kills_filename(game_num), error_bad_lines=False)
    kill_list = list(kills_frame.T.to_dict().values())
    update_ticks(kill_list, last_tick)
    kills_map = collections.defaultdict(list)
    for kill_data in kill_list:
      kills_map[kill_data['tick']].append(kill_data)

    # update last tick for next game offset
    last_tick = tick_list[-1]['tick']

    # for each kill, obtain all surrounding ticks for killer
    deltas = []
    for kill_data in kill_list:
      # when did the kill happen and by who? make sure to consider suicides
      kill_tick = kill_data['tick']
      killer_id = kill_data['killerId']
      kill_deltas = []

      if killer_id == -1:
        print(f'Death at tick {kill_tick} is suicide, skipping...')
        continue
      
      # get all surrounding ticks
      range_start = kill_tick  - KILL_PRIOR_NUM_TICKS
      range_end = kill_tick + KILL_POST_NUM_TICKS + 1
      tick_range = range(kill_tick  - KILL_PRIOR_NUM_TICKS, kill_tick + KILL_POST_NUM_TICKS + 1)
      surrounding_ticks = []
      for tick in tick_range:
        tick_player_data = ticks_map[tick]
        if killer_id in tick_player_data:
          surrounding_ticks.append(tick_player_data[killer_id])

      # sample once every KILL_TICK_GRANULARITY ticks to compute deltas
      sampled_ticks = []
      for i in range(0, len(surrounding_ticks), KILL_TICK_GRANULARITY):
        sampled_ticks.append(surrounding_ticks[i])

      if len(sampled_ticks) == 0:
        print(f'Kill at tick {kill_tick} had no ticks, skipping...')
        continue

      # compute deltas based on sampled ticks
      for i in range(1, len(sampled_ticks)):
        first = sampled_ticks[i - 1]
        second = sampled_ticks[i]
        delta = {}
        for k, v in first.items():
          if k in NON_DELTA_FIELDS:
            delta[k] = second[k]
          elif k in CUSTOM_DIFFS:
            delta[k] = CUSTOM_DIFFS[k](first[k], second[k])
          elif isinstance(v, int) or isinstance(v, float):
            delta[k] = second[k] - first[k]
          else:
            delta[k] = second[k]
        kill_deltas.append(delta)

      # compute aggregates of deltas for this kill
      aggregates = {}
      for d in kill_deltas:
        num_deltas = len(kill_deltas)
        for k, v in d.items():
          if is_number(v):
            max_key = f'{k}MaxDelta'
            min_key = f'{k}MinDelta'
            avg_key = f'{k}AvgDelta'
            aggregates[max_key] = max([e[k] for e in kill_deltas])
            aggregates[min_key] = min([e[k] for e in kill_deltas])
            aggregates[avg_key] = sum([e[k] for e in kill_deltas]) / num_deltas
        for k, v in aggregates.items():
          d[k] = v

      # add kill deltas to full list
      deltas += kill_deltas

    # write out deltas to new CSV
    if len(deltas) == 0:
      print(f'Game {game_num} has no deltas, skipping...')
      continue

    # append this game's results to the output file
    with open(output_filename(), 'a+') as output_file:
      writer = csv.DictWriter(output_file, deltas[0].keys())
      writer.writeheader()
      writer.writerows(deltas)

    print(f'Preprocessed game {game_num}.')

  print(f'Finished preprocessing {NUM_GAMES} games of {"test" if TEST else "training"} data.')


if __name__ == '__main__':
  main()
