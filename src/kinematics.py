import numpy as np
from scipy import signal

def time_to_reach(player_pos, player_vel, target_pos, max_accel=10.0, max_speed=22.0, diagnostics=None):
    """
    Calculates minimum time for a player to reach a target position.
    Uses kinematic equations accounting for current velocity, acceleration, and max speed.
    """
    # Vector from player to target
    displacement = target_pos - player_pos
    distance = np.linalg.norm(displacement)
    
    if distance < 0.1:  # Already at target
        if diagnostics is not None:
            diagnostics['player on top of you'] += 1
        return 0.0
        
    # Unit vector toward target
    direction = displacement / distance
    
    # Current speed and velocity component toward target
    current_speed = np.linalg.norm(player_vel)
    v_parallel = np.dot(player_vel, direction)  # Velocity toward target
    
    # Simplified model: assume player can instantly redirect velocity toward target
    t_brake = 0.0
    effective_distance = distance        
    # If already at or above max speed
    if v_parallel < 0:
        # Player is moving away from the target. They must brake to 0 first.
        # v = v0 + at => t = abs(v) / a
        t_brake = abs(v_parallel) / max_accel
        
        # Distance drifted away while braking: d = 0.5 * v * t
        drift_distance = 0.5 * abs(v_parallel) * t_brake
        
        # Now they have further to go, and they are starting from a standstill
        effective_distance += drift_distance
        v_initial = 0.0
    else:
        v_initial = v_parallel
        
    # Already moving at or above max speed TOWARD the target
    if v_initial >= max_speed:
        return (effective_distance / v_initial) + t_brake
        
    # Distance needed to reach max speed from current v_initial
    accel_distance = (max_speed**2 - v_initial**2) / (2 * max_accel)
    
    if accel_distance >= effective_distance:
        # Won't reach max speed, solve quadratic: 0.5*a*t^2 + v0*t - d = 0
        a = 0.5 * max_accel
        b = v_initial
        c = -effective_distance
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            if diagnostics is not None:
                diagnostics['err_time_to_reach_imaginary_root'] += 1
            return (effective_distance / max(1.0, v_initial)) + t_brake
            
        t_accel = (-b + np.sqrt(discriminant)) / (2*a)
        return t_accel + t_brake
        
    else:
        # Accelerate to max speed, then cruise
        t_accel = (max_speed - v_initial) / max_accel
        remaining_distance = effective_distance - accel_distance
        t_cruise = remaining_distance / max_speed
        
        return t_brake + t_accel + t_cruise

def get_player_physics(game, frame_number):
    """
    Calculates kinematic properties (velocity and acceleration) for all players.
    Uses a Savitzky-Golay filter over the past 1 second (25 frames).
    Results are cached on the game object to avoid redundant calculations.
    """
    if not hasattr(game, '_physics_cache'):
        game._physics_cache = {}
        
    if frame_number in game._physics_cache:
        return game._physics_cache[frame_number]
        
    # We need at least enough frames for a Savitzky-Golay window
    window_size = 5
    poly_order = 3
    
    # Look back up to 25 frames (~1 second)
    start_frame = max(0, frame_number - 25)
    
    if frame_number - start_frame < window_size:
        game._physics_cache[frame_number] = {}
        return {} 
        
    times = []
    player_trajectories = {} 
    
    # Fetch all moments at once to minimize pandas iloc overhead
    # We slice the dataframe from start_frame to frame_number
    moments_slice = game.moments.iloc[start_frame:frame_number + 1]
    
    for _, moment in moments_slice.iterrows():
        times.append(moment["universe_time"]/1000.0)
        
        frame_pids = set()
        for p in moment.positions:
            team_id, pid, x, y, radius = p
            if pid == -1: continue # Skip ball
            
            frame_pids.add(pid)
            if pid not in player_trajectories:
                # Backfill with current position if we catch them late
                player_trajectories[pid] = {'x': [x] * len(times[:-1]), 
                                            'y': [y] * len(times[:-1])}
            
            player_trajectories[pid]['x'].append(x)
            player_trajectories[pid]['y'].append(y)
            
        # Handle ghosting players
        for pid in player_trajectories:
            if pid not in frame_pids:
                last_x = player_trajectories[pid]['x'][-1]
                last_y = player_trajectories[pid]['y'][-1]
                player_trajectories[pid]['x'].append(last_x)
                player_trajectories[pid]['y'].append(last_y)

    times = np.array(times)
    if len(times) < window_size:
        game._physics_cache[frame_number] = {}
        return {}
        
    dt_avg = np.mean(np.diff(times))
    if dt_avg <= 0:
        game._physics_cache[frame_number] = {}
        return {}
        
    physics = {}
    
    final_moment = moments_slice.iloc[-1]
    pid_to_index = {p[1]: idx for idx, p in enumerate(final_moment.positions) if p[1] != -1}

    for pid, coords in player_trajectories.items():
        if pid not in pid_to_index:
            continue
            
        idx = pid_to_index[pid]
        
        try:
            vx = signal.savgol_filter(coords['x'], window_length=window_size, polyorder=poly_order, deriv=1, delta=dt_avg)
            vy = signal.savgol_filter(coords['y'], window_length=window_size, polyorder=poly_order, deriv=1, delta=dt_avg)
            
            ax = signal.savgol_filter(coords['x'], window_length=window_size, polyorder=poly_order, deriv=2, delta=dt_avg)
            ay = signal.savgol_filter(coords['y'], window_length=window_size, polyorder=poly_order, deriv=2, delta=dt_avg)
            
            v_curr = np.array([vx[-1], vy[-1]])
            a_curr = np.array([ax[-1], ay[-1]])
            
        except Exception as e:
            game.physics_diagnostics[f'err_savgol: {str(e)}'] += 1
            v_curr = np.array([0.0, 0.0])
            a_curr = np.array([0.0, 0.0])
        
        physics[idx] = {
            'velocity': v_curr,
            'acceleration': a_curr,
            'speed': np.linalg.norm(v_curr),
            'accel_mag': np.linalg.norm(a_curr)
        }

        game.physics_diagnostics['success_frames_processed'] += 1
    
    game._physics_cache[frame_number] = physics
    return physics
