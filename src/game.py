"""
Library for retrieving basketball player-tracking and play-by-play data.
"""

import os
import warnings
import json
import time
import urllib.request
import py7zr
import pandas as pd
import shutil
import sys
import subprocess
import socket

# Prevent urllib.request from hanging infinitely if GitHub drops a connection
socket.setdefaulttimeout(60.0) 
from subprocess import Popen, PIPE
import numpy as np
from scipy import spatial, integrate, signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc, Polygon
import seaborn as sns
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from collections import Counter

if not os.path.exists('temp'):
    os.makedirs('temp', exist_ok=True)


def _download_with_retry(url, dest, retries=3, backoff=2.0, verbose=True):
    """
    Download `url` to `dest`, retrying with exponential backoff.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  [download attempt {attempt}/{retries} failed] {url}: {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_err


class Game(object):
    """
    Class for basketball game.
    Contains play by play and player tracking data and methods for
    analysis and plotting.
    """

    def __init__(self, date, team1, team2, game_7z=None, temp_dir='temp', verbose=True,
                 tracking_url=None):
        """
        Args:
            date (str): 'MM.DD.YYYY', date of game
            team1 (str): 'XXX', abbreviation of team1 in data
                tracking file name (Home Team usually)
            team2 (str): 'XXX', abbreviation of team2 in data
                tracking file name (Away Team usually)
            game_7z (str): optional, the full name of the 7z file (e.g. '01.13.2016.GSW.at.DEN.7z')
            temp_dir (str): directory to download and extract tracking data for processing
            verbose (bool): whether to print status messages
            tracking_url (str): optional, full URL to the tracking .7z. Overrides the
                default sealneaward mirror — use it to pull a game from an alternate
                source (e.g. the linouk23 repo) when the default mirror's copy is
                missing or corrupt. The play-by-play events are still fetched from the
                sealneaward mirror keyed on the game_id parsed from the tracking json.

        Attributes:
            date (str): 'MM.DD.YYYY', date of game
            team1 (str): 'XXX', abbreviation of team1 in data
                tracking file name
            team2 (str): 'XXX', abbreviation of team2 in data
                tracking file name
            tracking_id (str): id to access player tracking data
                Due to the way the SportVU data is stored, game_id is
                complicated: 'MM.DD.YYYY.AWAYTEAM.at.HOMETEAM'
                For Example: 01.13.2016.GSW.at.DEN
            tracking_data (dict): Dictionary of unstructured tracking
                data scraped from github.
            game_id (str): ID for game.  Luckily, SportVU and play by
                play use the same game ID
            pbp (pd.DataFrame): Play by play data.  33 columns per pbp
                instance.
            moments (pd.DataFrame): DataFrame of player tracking data.
                Each entry is a single snap-shot of where the players
                are at a given time on the court.
                Columns: ['quarter', 'universe_time', 'quarter_time',
                'shot_clock', 'positions', 'game_time'].
                moments['positions'] contains a list of where each player
                and the ball are located.
            player_ids (dict): dictionary of {player: player_id} for
                all players in game.
            away_id (int): ID of away team
            home_id (int): ID of home team
            team_colors (dict): dictionary of colors for each team and
                ball. Used for plotting.
            home_team (str): 'XXX', abbreviation of home team
            away_team (str): 'XXX', abbreviation of away team
        """
        self.date = date
        self.team1 = team1
        self.team2 = team2
        self.flip_direction = False
        
        self.verbose = verbose
        
        if game_7z:
            self.tracking_id = game_7z.replace('.7z', '')
        else:
            # Traditional fallback
            self.tracking_id = ('{self.date}.{self.team2}.at.{self.team1}'
                                .format(self=self))
        
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.datalink = tracking_url or f"https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/{self.tracking_id}.7z"
        self.tracking_data = None
        self.game_id = None
        self.pbp = None
        self.moments = None
        self.player_ids = None
        self._get_tracking_data()
        self._get_playbyplay_data()
        self._format_tracking_data()
        self._get_player_ids()
        self._get_player_jerseys()
        self.away_id = self.tracking_data['events'][0]['visitor']['teamid']
        self.home_id = self.tracking_data['events'][0]['home']['teamid']
        self.home_team = (self.tracking_data['events'][0]['home']
                          ['abbreviation'])
        self.away_team = (self.tracking_data['events'][0]['visitor']
                          ['abbreviation'])
        # Resolve team colors from the NBA palette; fall back to red/blue
        # if a team abbreviation isn't recognized.
        from visualization import get_team_color
        self.team_colors = {-1: "orange",
                            self.away_id: get_team_color(self.away_team, "blue"),
                            self.home_id: get_team_color(self.home_team, "red")}
        self.flip_direction = False
        ##self._determine_direction()
        self.physics_diagnostics = Counter()
        
        if self.verbose:
            print('All data is loaded')

    def _get_tracking_data(self):
        """
        Helper function for retrieving tracking data
        Tracking Data is provided by NBA.com,
        hosted at: https://www.github.com/neilmj
        Update it is hosted now on https://github.com/sealneaward/nba-movement-data/tree/master
        """
        # Retrieve and extract Data into temp folder

        # Download the 7z file using urllib (cross-platform)
        if self.verbose:
            print(f"Downloading data from {self.datalink}...")
        try:
            _download_with_retry(self.datalink, f"{self.temp_dir}/game.7z", verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(f"Error downloading {self.datalink}: {e}")
            raise
            
        if self.verbose:
            print("Download complete. Extracting...")
        
        # Extract using py7zr (pure Python implementation)
        with py7zr.SevenZipFile(f"{self.temp_dir}/game.7z", mode='r') as archive:
            archive.extractall(path=self.temp_dir)
            
        if self.verbose:
            print("Extraction complete.")
        
        # Wait a moment for file handles to be released (Windows issue)
        time.sleep(0.5)
        
        # Try to remove the 7z file with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.remove(f"{self.temp_dir}/game.7z")
                break
            except (PermissionError, FileNotFoundError):
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"Warning: Could not delete {self.temp_dir}/game.7z. You may need to delete it manually.")
        
        # Extract game ID from extracted file name.
        for file in os.listdir(self.temp_dir):
            if os.path.splitext(file)[1] == '.json':
                self.game_id = file[:-5]

        # Load tracking data and remove json file
        with open(f"{self.temp_dir}/{self.game_id}.json") as data_file:
            self.tracking_data = json.load(data_file)  # Load this json
        os.remove(f"{self.temp_dir}/{self.game_id}.json")
        return self

    def _get_playbyplay_data(self):
        """
        Helper function for retrieving play-by-play data.
        Play-by-play data is obtained via API call to NBA.com
        This service is likely to go down at any moment and ruin this
        whole project.
        """
        # Download play-by-play data from GitHub
        pbp_link = f"https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/events/{self.game_id}.csv"
        pbp_filename = f"{self.temp_dir}/{self.game_id}_events.csv"
        
        if self.verbose:
            print(f"Downloading play-by-play data from {pbp_link}...")

        _download_with_retry(pbp_link, pbp_filename, verbose=self.verbose)

        if self.verbose:
            print("Play-by-play download complete.")
        
        # load play by play into pandas DataFrame
        self.pbp = pd.read_csv(pbp_filename)
        os.remove(pbp_filename)

        # Get time in quarter remaining to cross-reference tracking data
        self.pbp['Qmin'] = (self.pbp['PCTIMESTRING'].str
                            .split(':', expand=True)[0])
        self.pbp['Qsec'] = (self.pbp['PCTIMESTRING'].str
                            .split(':', expand=True)[1])
        self.pbp['Qtime'] = (self.pbp['Qmin'].astype(int)*60 +
                             self.pbp['Qsec'].astype(int))
        self.pbp['game_time'] = ((self.pbp['PERIOD'] - 1) * 720 +
                                 (720 - self.pbp['Qtime']))

        # Format score so that it makes sense: 'XX-XX'
        self.pbp['SCORE'] = (self.pbp['SCORE']
                             .ffill()
                             .fillna('0 - 0'))
                             
        def parse_margin(m):
            if pd.isna(m): return pd.NA
            if m == 'TIE': return 0
            try: return int(m)
            except: return pd.NA
            
        filled_margins = (
            self.pbp['SCOREMARGIN']
            .apply(parse_margin)
            .astype('Int64')
            .ffill()
            .fillna(0)
            .shift(1)
            .fillna(0)
        )

        self.pbp['PRE_PLAY_MARGIN'] = filled_margins
        
        return self

    def _get_player_ids(self):
        """
        Helper function for returning player ids for all players in game.
        Note: This data may also be somewhere more conveniently
            accessible in tracking_data.
        """
        ids = {}
        for index, row in self.pbp.iterrows():
            if row['PLAYER1_NAME'] not in ids:
                ids[row['PLAYER1_NAME']] = row['PLAYER1_ID']
            if row['PLAYER2_NAME'] not in ids:
                ids[row['PLAYER2_NAME']] = row['PLAYER2_ID']
            if row['PLAYER3_NAME'] not in ids:
                ids[row['PLAYER3_NAME']] = row['PLAYER3_ID']
        ids.pop(None, None)  # Remove None key if it exists, otherwise do nothing
        self.player_ids = ids
        return self

    def _get_player_jerseys(self):
        """
        Helper function for returning player jerseys for all players in game.
        """
        jerseys = {}
        # Combine home and away players
        # The structure is assumed to be self.tracking_data['events'][0]['home']['players'] based on standard SportVU data
        players = (self.tracking_data['events'][0]['home']['players'] +
                   self.tracking_data['events'][0]['visitor']['players'])
        for player in players:
            jerseys[player['playerid']] = player['jersey']
        self.player_jerseys = jerseys
        return self

    def _format_tracking_data(self):
        """
        Helper function to format tracking data into pandas DataFrame
        """
        events = pd.DataFrame(self.tracking_data['events'])
        moments = []
        # Extract 'moments': Each moment is an individual frame
        for row in events['moments']:
            for inner_row in row:
                moments.append(inner_row)
        moments = pd.DataFrame(moments)
        moments = moments.drop_duplicates(subset=[1])
        moments = moments.reset_index()

        moments.columns = ['index', 'quarter', 'universe_time', 'quarter_time',
                           'shot_clock', 'unknown', 'positions']
        moments['game_time'] = (moments.quarter - 1) * 720 + \
                               (720 - moments.quarter_time)
        moments.drop(['index', 'unknown'], axis=1, inplace=True)
        self.moments = moments
        return self


    def _get_commentary(self, game_time, commentary_length=6,
                        commentary_depth=10):
        """
        Helper function for returning play by play events for a
        given game time.

        Args:
            game_time (int): game time (in seconds) for which to
                retrieve commentary for
            commentary_length (int): Number of play-by-play calls to
                include in commentary
            commentary_depth (int): Number of seconds to look in past
                to retrieve play-by-play calls
                commentary_depth=10 looks at previous 10 seconds of
                game for play-by-play calls

        Returns: tuple of information (commentary_script, score)
            commentary_script (str): string of commentary
                Most recent play-by-play calls, separated by line breaks
            score (str): Score at current time 'XX - XX'
        """
        commentary = []
        score = "0 - 0"
        for game_second in range(game_time - commentary_depth, game_time + 2):
            for index, row in self.pbp[self.pbp.game_time ==
                                       game_second].iterrows():
                # Check if descriptions are strings (not NaN)
                if isinstance(row['HOMEDESCRIPTION'], str):
                    commentary.append('{self.home_team}: '.format(self=self) +
                                      str(row['HOMEDESCRIPTION']))
                if isinstance(row['VISITORDESCRIPTION'], str):
                    commentary.append('{self.away_team}: '.format(self=self) +
                                      str(row['VISITORDESCRIPTION']))
                if isinstance(row['NEUTRALDESCRIPTION'], str):
                    commentary.append(str(row['NEUTRALDESCRIPTION']))
                score = str(row['SCORE'])
        
        # Pad with empty strings if fewer than commentary_length items
        while len(commentary) < commentary_length:
            commentary.append("")
            
        # Take only the last commentary_length items if we have too many
        if len(commentary) > commentary_length:
            commentary = commentary[-commentary_length:]
            
        commentary_script = "\n".join(commentary)
        return (commentary_script, score)

    def _get_player_actions(self, player_name, action):
        """
        Helper function to get all times a player performed a specific action

        Args:
            player_name (str): name of player to get all actions for
            action {'all_FG', 'made_FG', 'miss_FG', 'rebound'}:
                Type of action to get all times for.

        Returns:
            times (list): list of game times a player performed a
                specific specific action
        """
        player_id = self.player_ids[player_name]
        action_dict = {'all_FG': [1, 2], 'made_FG': [1],
                       'miss_FG': [2], 'rebound': [4]}
        action_df = self.pbp[(self.pbp['PLAYER1_ID'] == player_id) &
                             (self.pbp['EVENTMSGTYPE']
                              .isin(action_dict[action]))]
        times = list(action_df['game_time'])
        return times

    def _get_moment_details(self, frame_number, highlight_player=None):
        """
        Helper function for getting important information for a given frame

        Args:
            frame_number (int): Frame in game to retrieve data for
                frame_number gets player tracking data from
                    moments.iloc[frame_number]
            highlight_player (str): Name of player to be highlighted
                in downstream plotting.
                if None, no player is highlighted.

        Returns: tuple of data
            game_time (int): seconds into game of current moment
            x_pos (list): list of x coordinants for all players and ball
            y_pos (list): list of y coordinants for all players and ball
            colors (list): color coding of each player/ball for coordinant data
            sizes (list): size of each player/ball
                (used for showing ball height)
            quarter (int): Game quarter
            shot_clock (str): shot clock
            game_clock (str): game clock
            edges (list): list of marker edge sizes of each player for video.
                useful when trying to highlight a player by making
                their edge thicker.
            universe_time (int): Time in the universe, in msec
        """
        current_moment = self.moments.iloc[frame_number]
        game_time = int(np.round(current_moment['game_time']))
        universe_time = int(current_moment['universe_time'])
        x_pos, y_pos, colors, sizes, edges, jerseys = [], [], [], [], [], []
        # Get player positions
        for player in current_moment.positions:
            x_pos.append(player[2])
            y_pos.append(player[3])
            colors.append(self.team_colors[player[0]])
            # Use ball height for size (useful to see a shot)
            if player[0] == -1:
                sizes.append(max(150 - 2*(player[4] - 5)**2, 10))
            else:
                sizes.append(200)
            # highlight_player makes their outline much thicker on the video
            if (highlight_player and
                    player[1] == self.player_ids[highlight_player]):
                edges.append(5)
            else:
                edges.append(0.5)
            # Add jersey number
            if player[1] != -1:
                jerseys.append(self.player_jerseys[player[1]])
            else:
                jerseys.append(None)
        # Unfortunately, the plot is below the y axis,
        # so the y positions need to be corrected
        y_pos = np.array(y_pos) - 50
        shot_clock = current_moment.shot_clock
        if np.isnan(shot_clock):
            shot_clock = 24.00
        shot_clock = str(shot_clock).split('.')[0]
        game_min, game_sec = divmod(current_moment.quarter_time, 60)
        game_clock = "%02d:%02d" % (game_min, game_sec)
        quarter = current_moment.quarter
        return (game_time, x_pos, y_pos, colors, sizes, quarter,
                shot_clock, game_clock, edges, universe_time, jerseys)


    def _in_formation(self, frame_number):
        """
        This is a complicated method to explain, but it is actually
        very simple.
        It determines if the game is in a set offense/defense.
        It basically returns True if a normal play is being run,
        and False if the game is in transition, out of bounds,
        free throw, etc.  It is useful for analyzing plays that teams
        run, and discarding all extranous times from the game.
        """
        # Get relevant moment details
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        shot_clock = details[6]
        # Determine if offense/defense is set
        if float(shot_clock) < 23:
            if (x_pos < 47).all() or (x_pos > 47).all():
                return True
        return False


    def get_frame(self, game_time):
        """
        Converts a game time to a frame number.
        Finds the closest available frame index to the requested game_time.
        Replaced dangerous `while True` countdown loop with vectorized lookup.

        Args:
            game_time (float): game time in seconds of interest

        Returns:
            frame (int): closest available frame index
        """
        if self.moments is None or len(self.moments) == 0:
            raise ValueError("No moments data available")
            
        # Find index with smallest absolute difference to requested game_time
        diffs = np.abs(self.moments['game_time'] - game_time)
        closest_idx = diffs.idxmin()
        
        return closest_idx

    def print_physics_diagnostics(self):
        """Prints a summary of the physics engine's health."""
        print("\n=== Physics Engine Diagnostics ===")
        total_success = self.physics_diagnostics.get('success_frames_processed', 0)
        
        total_errors = sum(count for key, count in self.physics_diagnostics.items() if key.startswith('err_'))
        
        print(f"Frames Processed Successfully: {total_success}")
        print(f"Total Computation Errors: {total_errors}")
        print("-" * 30)
        
        for key, count in self.physics_diagnostics.items():
            if key.startswith('err_'):
                print(f"  [!] {key}: {count} occurrences")
        print("==================================\n")


    def get_play_frames(self, event_num, play_type='offense'):
        """
        Args:
            event_num (int): EVENTNUM of interest in games.pbp
                NOTE: Check pbpevents.txt for event numbers
            play_type (str in ['offense', 'defense']): Team of interest
                is offense or defense

        Returns:
            tuple of (start_time (int), end_time (int)): start time
                and end time in seconds for play of interest
        """
        play_index = self.pbp[self.pbp['EVENTNUM'] == event_num].index[0]
        event_team = str(self.pbp[self.pbp['EVENTNUM'] == event_num]
                         .PLAYER1_TEAM_ABBREVIATION.head(1).values[0])
        if event_team == self.home_team:
            target_team = 'home'
        if event_team == self.away_team:
            target_team = 'away'
        end_time = int(self.pbp[self.pbp['EVENTNUM'] == event_num].game_time)
        # To find lower bound on starting frame of the play,
        # determining when previous play ended
        putative_start_time = int(self.pbp.iloc[play_index-1].game_time)
        putative_start_frame = self.get_frame(putative_start_time)
        end_frame = self.get_frame(end_time)
        for test_frame in range(putative_start_frame, end_frame):
            if self.get_offensive_team(test_frame) == target_team:
                break
        # If the previous loop never found an offensive play,
        # the function returns None
        else:
            return None
        # Add two seconds to game time to let the players settle into position
        start_frame = self.get_frame(round(self.moments.iloc[test_frame].game_time + 2))
        return (start_frame, end_frame)
