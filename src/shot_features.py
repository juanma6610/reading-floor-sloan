"""
Shot Feature Extraction Pipeline

Extracts, for every shot attempt in a game,  engineered features from
SportVU tracking data (geometry, defender pressure, shooter/defender
kinematics, release mechanics, tempo, spacing) plus 8 soft archetype
probabilities (4 shooter + 4 defender) from pre-computed GMM clustering,
"""

import numpy as np
import pandas as pd
import traceback
from scipy.signal import find_peaks

from game import Game
from kinematics import get_player_physics
from spatial import get_spacing_area

# ============================================================
# Constants
# ============================================================
LEFT_BASKET = np.array([5.25, 25.0])
RIGHT_BASKET = np.array([88.75, 25.0])
THREE_POINT_DIST = 23.75

def load_archetypes(shooter_path='clusters/gmm_soft_labels_15_16.csv',
                    defender_path='clusters/gmm_soft_labels_def_15_16.csv'):
    shooter_df = pd.read_csv(shooter_path)
    defender_df = pd.read_csv(defender_path)
    
    shooter_cols = [col for col in shooter_df.columns if col != 'Player']
    defender_cols = [col for col in defender_df.columns if col != 'Player']
    
    shooter_baseline = shooter_df[shooter_cols].mean().to_dict()
    defender_baseline = defender_df[defender_cols].mean().to_dict()
    
    return shooter_df, defender_df, shooter_cols, defender_cols, shooter_baseline, defender_baseline

def _get_nearest_basket(shooter_x):
    if shooter_x < 47:
        return LEFT_BASKET
    return RIGHT_BASKET

def _decompose_velocity(velocity, shot_direction):
    par_vel = np.dot(velocity, shot_direction)
    perp_vel = np.linalg.norm(velocity - par_vel * shot_direction)
    return par_vel, perp_vel

class ShotFeatureExtractor:
    def __init__(self, game, shooter_map, defender_map, s_base, d_base, verbose=True):
        self.game = game
        self.shooter_map = shooter_map
        self.defender_map = defender_map
        self.s_base = s_base
        self.d_base = d_base
        self.verbose = verbose

    def _get_shot_events(self):
        shots = self.game.pbp[self.game.pbp['EVENTMSGTYPE'].isin([1, 2])].copy()
        shots = shots[shots['PLAYER1_NAME'].notna()]

        adjusted_margins = np.where(
            shots['PLAYER1_TEAM_ID'] == self.game.home_id,
            self.game.pbp.loc[shots.index, 'PRE_PLAY_MARGIN'],
            self.game.pbp.loc[shots.index, 'PRE_PLAY_MARGIN'] * -1
        )

        combined_desc = shots["HOMEDESCRIPTION"].fillna("") + " " + shots["VISITORDESCRIPTION"].fillna("")
        shots["description"] = combined_desc.str.strip().str.replace("'", " ").str.replace('"', ' ').str.replace('\n', ' ').str.replace('\r', ' ')

        shot_events = pd.DataFrame({
            'game_time': shots['game_time'],
            'player_name': shots['PLAYER1_NAME'],
            'player_id': shots['PLAYER1_ID'],
            'made': (shots['EVENTMSGTYPE'] == 1).astype(int),
            'quarter': shots['PERIOD'],
            'score_margin': adjusted_margins.astype(int),
            "description": shots['description']
        })
        
        return shot_events.reset_index(drop=True)

    def _find_shot_release_frame(self, pbp_frame, shooter_id, max_lookback_frames=200):
        start_search = max(0, pbp_frame - max_lookback_frames)
        end_search = min(len(self.game.moments), pbp_frame + 25)
        
        slice_df = self.game.moments.iloc[start_search:end_search]
        
        ball_z_history = []
        frame_indices = []
        for i, moment in slice_df.iterrows():
            z_val = -1.0
            for p in moment.positions:
                if p[1] == -1:
                    z_val = p[4]
                    break
            ball_z_history.append(z_val)
            frame_indices.append(i)
            
        ball_z_history = np.array(ball_z_history)
        
        peaks, _ = find_peaks(ball_z_history, height=9.0)
        
        valid_release_frame = -1
        
        for apex_idx in reversed(peaks):
            apex_frame = frame_indices[apex_idx]
            
            # Look backwards from peak
            for f in range(apex_frame, start_search, -1):
                try:
                    moment = self.game.moments.iloc[f]
                    ball_data, shooter_data = None, None
                    
                    for p in moment.positions:
                        if p[1] == -1:
                            ball_data = {'x': p[2], 'y': p[3], 'z': p[4]}
                        elif p[1] == shooter_id:
                            shooter_data = {'x': p[2], 'y': p[3]}
                            
                    if ball_data and shooter_data:
                        dist = np.hypot(ball_data['x'] - shooter_data['x'], 
                                        ball_data['y'] - shooter_data['y'])
                        
                        if ball_data['z'] <= 10.0 and dist <= 2.5:
                            valid_release_frame = f
                            break
                except Exception:
                    continue
                    
            if valid_release_frame != -1:
                break

        if valid_release_frame != -1:
            return valid_release_frame, self.game.moments.iloc[valid_release_frame].shot_clock
        else:
            if self.verbose:
                print(f"  [X] Dropping corrupted tracking data for shooter {shooter_id}.")
            return None, None

    def _check_catch_and_shoot(self, shot_frame, shooter_id, fps=25, max_touch_seconds=12.0):
        max_frames_back = int(fps * max_touch_seconds)
        start_frame = max(0, shot_frame - max_frames_back)
        
        catch_frame = None
        has_dribbled = False
        consecutive_far_frames = 0
        
        slice_df = self.game.moments.iloc[start_frame:shot_frame+1][::-1]
        
        for f, moment in slice_df.iterrows():
            ball_data = None
            shooter_data = None
            
            for p in moment.positions:
                if p[1] == -1:
                    ball_data = {'x': p[2], 'y': p[3], 'z': p[4]}
                elif p[1] == shooter_id:
                    shooter_data = {'x': p[2], 'y': p[3]}
                    
            if not ball_data or not shooter_data:
                continue 
                
            if ball_data['z'] < 1.5:
                has_dribbled = True
                
            dist = np.hypot(ball_data['x'] - shooter_data['x'], 
                            ball_data['y'] - shooter_data['y'])
            
            if dist > 5.0:
                consecutive_far_frames += 1
                if consecutive_far_frames >= 3:
                    catch_frame = f + 3 
                    break
            else:
                consecutive_far_frames = 0
                
        if catch_frame is None:
            touch_time = float(max_touch_seconds)
        else:
            frames_held = shot_frame - catch_frame
            touch_time = max(0.0, frames_held / float(fps))
            
        is_cs = 1 if (not has_dribbled and touch_time <= 2.0) else 0

        return is_cs, touch_time

    def _get_release_mechanics(self, frame, flipped_court, fps=25):
        """
        Release-point physics from the ball trajectory around the release frame.

        Returns
        -------
        release_height : float    Ball z at release (ft).
        release_speed  : float    3D ball speed at release (ft/s), 3-frame central diff.
        release_angle  : float    Vertical launch angle (degrees above horizontal).
        release_x      : float    Ball x at release, court-flipped to match feature space.
        release_y      : float    Ball y at release, court-flipped.

        All NaN on failure (start/end of game, ball missing in any of the three frames).
        """
        nan_pack = (np.nan, np.nan, np.nan, np.nan, np.nan)

        if frame <= 0 or frame >= len(self.game.moments) - 1:
            return nan_pack

        def _ball_xyz(f):
            try:
                moment = self.game.moments.iloc[f]
                for p in moment.positions:
                    if p[1] == -1:  # ball
                        return np.array([p[2], p[3], p[4]], dtype=float)
            except Exception:
                pass
            return None

        pos_prev = _ball_xyz(frame - 1)
        pos_now  = _ball_xyz(frame)
        pos_next = _ball_xyz(frame + 1)

        if pos_prev is None or pos_now is None or pos_next is None:
            return nan_pack

        # 3-frame central difference: dt across 2 frames = 2 / fps  (0.08s @ 25fps)
        velocity = (pos_next - pos_prev) / (2.0 / fps)

        speed_3d = float(np.linalg.norm(velocity))
        speed_xy = float(np.hypot(velocity[0], velocity[1]))

        if speed_xy > 1e-6:
            release_angle = float(np.degrees(np.arctan2(velocity[2], speed_xy)))
        else:
            release_angle = 90.0 if velocity[2] > 0 else -90.0

        release_height = float(pos_now[2])

        # Apply the same court flip used elsewhere so release xy lines up with shooter xy.
        rx, ry = float(pos_now[0]), float(pos_now[1])
        if flipped_court:
            rx = 94.0 - rx
            ry = 50.0 - ry

        return release_height, speed_3d, release_angle, rx, ry

    def extract_all(self):
        self.game.tracking_data = None # Free memory
        
        shot_events = self._get_shot_events()
        
        if self.verbose:
            print(f"Found {len(shot_events)} shot attempts in game")
        
        all_features = []
        skipped = 0
        
        for idx, shot in shot_events.iterrows():
            try:
                features = self._extract_single_shot(shot)
                if features is not None:
                    all_features.append(features)
                else:
                    skipped += 1
            except Exception as e:
                if self.verbose:
                    print(f"Error processing shot: {e}")
                    traceback.print_exc()  
                skipped += 1
        
        if self.verbose:
            print(f"Successfully extracted {len(all_features)} shots, skipped {skipped}")
        
        return pd.DataFrame(all_features)

    def _extract_single_shot(self, shot):
        game_time = shot['game_time']
        player_id = shot['player_id']
        player_name = shot['player_name']
        quarter = shot['quarter']
        
        try:
            pbp_frame = self.game.get_frame(game_time)
            frame, shot_clock = self._find_shot_release_frame(pbp_frame, player_id)
            if pd.isna(shot_clock) or shot_clock is None:
                shot_clock = game_time if game_time < 24.0 else 24.0
            if frame is None:
                return None 
        except Exception as e:
            if self.verbose: print(f"Exception getting frame: {e}")
            return None
            
        current_moment = self.game.moments.iloc[frame]
        
        players = {}
        for idx, p in enumerate(current_moment.positions):
            team_id, pid, x, y, radius = p
            if pid != -1:  
                players[pid] = {'team_id': team_id, 'x': x, 'y': y, 'index': idx}
                
        if player_id not in players:
            return None 
            
        shooter_team_id = players[player_id]['team_id']
        flipped_court = False
        
        shooter_x_raw = players[player_id]['x']
        basket = _get_nearest_basket(shooter_x_raw)
        
        if basket is LEFT_BASKET:
            flipped_court = True
            basket = RIGHT_BASKET
            for pid in players:
                players[pid]['x'] = 94.0 - players[pid]['x']
                players[pid]['y'] = 50.0 - players[pid]['y']
                
        shooter_pos = np.array([players[player_id]['x'], players[player_id]['y']])
        
        shot_vector = basket - shooter_pos
        shot_dist = np.linalg.norm(shot_vector)
        if shot_dist < 0.1 or shot_dist > 40.0: 
            return None
        
        shot_direction = shot_vector / shot_dist
        
        dist = shot_dist
        x = shooter_pos[0]
        y = shooter_pos[1]
        
        dy = shooter_pos[1] - basket[1]
        dx = shooter_pos[0] - basket[0]
        shot_angle = np.degrees(np.arctan2(dy, dx))
        
        defenders = {}
        def_distances_list = []
        
        for pid, data in players.items():
            if data['team_id'] != shooter_team_id:
                d_pos = np.array([data['x'], data['y']])
                d_dist = np.linalg.norm(d_pos - shooter_pos)
                defenders[pid] = {'pos': d_pos, 'dist': d_dist}
                def_distances_list.append(d_dist)
                
        if not defenders: return None
        
        sorted_def_ids = sorted(defenders.keys(), key=lambda k: defenders[k]['dist'])
        closest_def_id = sorted_def_ids[0]
        closest_def_pos = defenders[closest_def_id]['pos']
        closest_def_dist = defenders[closest_def_id]['dist']

        def_vector = closest_def_pos - shooter_pos
        def_vector_norm = np.linalg.norm(def_vector)
        if def_vector_norm > 0:
            def_direction = def_vector / def_vector_norm
            cos_def_angle = np.clip(np.dot(shot_direction, def_direction), -1.0, 1.0)
            closest_def_angle = np.degrees(np.arccos(cos_def_angle))
        else:
            closest_def_angle = 0.0
            
        def transpose_vector(vec):
            if vec is None or len(vec) < 2:
                return np.array([0.0, 0.0])
            if flipped_court: 
                return np.array([-vec[0], -vec[1]])
            return np.array(vec)

        physics_at_shot = get_player_physics(self.game, frame)
        shooter_idx = players[player_id]['index']
        closest_def_idx = players[closest_def_id]['index']
        
        lookback_frame = max(0, frame - 10)
        physics_lookback = get_player_physics(self.game, lookback_frame)

        if physics_at_shot and shooter_idx in physics_at_shot and physics_lookback and closest_def_idx in physics_at_shot:
            shooter_vel = transpose_vector(physics_at_shot[shooter_idx]['velocity'])
            shooter_acc = transpose_vector(physics_at_shot[shooter_idx]['acceleration'])
            
            shooter_par_vel, shooter_perp_vel = _decompose_velocity(shooter_vel, shot_direction)
            shooter_par_acc, shooter_perp_acc = _decompose_velocity(shooter_acc, shot_direction)
            
            def_vel = transpose_vector(physics_at_shot[closest_def_idx]['velocity'])
            def_acc = transpose_vector(physics_at_shot[closest_def_idx]['acceleration'])
            
            def_par_vel, def_perp_vel = _decompose_velocity(def_vel, shot_direction)
            def_par_acc, def_perp_acc = _decompose_velocity(def_acc, shot_direction)
        else:
            shooter_par_vel = shooter_perp_vel = shooter_par_acc = shooter_perp_acc = 0.0
            def_par_vel = def_perp_vel = def_par_acc = def_perp_acc = 0.0

        is_catch_and_shoot, touch_time = self._check_catch_and_shoot(frame, player_id)

        release_height, release_speed, release_angle, release_x, release_y = \
            self._get_release_mechanics(frame, flipped_court)

        if physics_lookback and closest_def_idx in physics_lookback:
            from kinematics import time_to_reach
            def_vel_closeout = transpose_vector(physics_lookback[closest_def_idx]['velocity'])
            time_to_contest = time_to_reach(closest_def_pos, def_vel_closeout, shooter_pos)
        else:
            time_to_contest = np.nan 
            self.game.physics_diagnostics['err_time_to_contest_fallback_nan'] += 1

        try:
            home_hull, away_hull = get_spacing_area(self.game, frame)
        except Exception:
            home_hull, away_hull = 0.0, 0.0
            
        offense_convex_hull = home_hull if shooter_team_id == self.game.home_id else away_hull
        defense_convex_hull = away_hull if shooter_team_id == self.game.home_id else home_hull
        ratio_off_def_hull = offense_convex_hull / (defense_convex_hull + 1e-5)

        def_distances_arr = np.array(def_distances_list)
        def_very_tight = int(np.sum(def_distances_arr <= 2.0))
        def_tight = int(np.sum((def_distances_arr > 2.0) & (def_distances_arr <= 4.0)))
        def_open = int(np.sum((def_distances_arr > 4.0) & (def_distances_arr <= 6.0)))

        if len(sorted_def_ids) >= 2:
            second_def_id = sorted_def_ids[1]
            second_def_idx = players[second_def_id]['index']
            second_closest_def_pos = defenders[second_def_id]['pos']
            second_closest_def_dist = defenders[second_def_id]['dist']
            
            if physics_lookback and second_def_idx in physics_lookback:
                from kinematics import time_to_reach
                second_def_vel = transpose_vector(physics_lookback[second_def_idx]['velocity'])
                second_closest_def_time = time_to_reach(second_closest_def_pos, second_def_vel, shooter_pos)
            else:
                second_closest_def_time = second_closest_def_dist / 15.0
        else:
            second_closest_def_dist = np.nan
            second_closest_def_time = np.nan

        is_3_pointer = 1 if dist >= THREE_POINT_DIST else 0
        
        s_probs = self.shooter_map.get(player_name, self.s_base) if self.shooter_map else self.s_base
        
        def _get_player_name(game, pid):
            for name, id_ in game.player_ids.items():
                if id_ == pid: return name
            return "Unknown Defender"
            
        closest_def_name = _get_player_name(self.game, closest_def_id)
        d_probs = self.defender_map.get(closest_def_name, self.d_base) if self.defender_map else self.d_base

        features = {
            'player_name': player_name,
            'game_time': game_time,
            'quarter': shot['quarter'],
            'score_margin': shot['score_margin'],
            "team_id": shooter_team_id,
            "game_id": self.game.game_id,
            "description": shot['description'],
            'closest_def_name': closest_def_name,
            'made_shot': shot['made'],
            'dist': dist,
            'x': x,
            'y': y,
            'shot_angle': shot_angle,
            'closest_def_dist': closest_def_dist,
            'closest_def_angle': closest_def_angle,
            'shot_clock': shot_clock,
            'shooter_par_vel': shooter_par_vel,
            'shooter_perp_vel': shooter_perp_vel,
            'def_par_vel': def_par_vel,
            'def_perp_vel': def_perp_vel,
            'shooter_par_acc': shooter_par_acc,
            'shooter_perp_acc': shooter_perp_acc,
            'def_par_acc': def_par_acc,
            'def_perp_acc': def_perp_acc,
            'time_to_contest': time_to_contest,
            'ratio_off_def_hull': ratio_off_def_hull,
            'touch_time': touch_time,                 
            'is_catch_and_shoot': is_catch_and_shoot,
            'second_closest_def_dist': second_closest_def_dist,
            'second_closest_def_time': second_closest_def_time,
            'is_3_pointer': is_3_pointer,
            'def_very_tight': def_very_tight,
            'def_tight': def_tight,
            'def_open': def_open,
            'release_height': release_height,
            'release_speed': release_speed,
            'release_angle': release_angle,
            'release_x': release_x,
            'release_y': release_y,
            **s_probs,
            **d_probs
        }
        
        return features

def extract_shot_features(game, shooter_map=None, defender_map=None, s_base=None, d_base=None, verbose=True):
    extractor = ShotFeatureExtractor(game, shooter_map, defender_map, s_base, d_base, verbose=verbose)
    return extractor.extract_all()

def get_feature_columns(df):
    metadata_cols = ['player_name', 'game_time', 'quarter', 'made_shot']
    return [c for c in df.columns if c not in metadata_cols]

if __name__ == '__main__':
    print("=" * 60)
    print("Shot Feature Extraction Pipeline")
    print("=" * 60)
    
    print("\n[1/3] Loading player archetypes...")
    s_df, d_df, s_cols, d_cols, s_base, d_base = load_archetypes()
    
    shooter_map = s_df.set_index('Player').to_dict('index')
    defender_map = d_df.set_index('Player').to_dict('index')

    print(f"  Loaded {len(shooter_map)} shooter archetypes")
    print(f"  Loaded {len(defender_map)} defender archetypes")

    print("\n[2/3] Loading game data (GSW @ DEN, 01.13.2016)...")
    game = Game('01.13.2016', 'DEN', 'GSW')
    
    print("\n[3/3] Extracting shot features...")
    df = extract_shot_features(
        game, shooter_map, defender_map,
        s_base, d_base, 
        verbose=True
    )
    
    output_path = 'data/shot_features.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} shots × {len(df.columns)} columns to {output_path}")
    
    print("\n" + "=" * 60)
    print("Feature Summary")
    print("=" * 60)
    feature_cols = get_feature_columns(df)
    print(f"  Input features: {len(feature_cols)}")
    print(f"  Shots extracted: {len(df)}")
    print(f"  Made shots: {df['made_shot'].sum()} ({df['made_shot'].mean()*100:.1f}%)")
    print(f"  Missed shots: {(1-df['made_shot']).sum():.0f} ({(1-df['made_shot']).mean()*100:.1f}%)")
    
    game.print_physics_diagnostics()
