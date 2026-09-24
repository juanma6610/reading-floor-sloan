import numpy as np
from scipy.spatial import ConvexHull, Voronoi

# Import kinematics if needed
from kinematics import time_to_reach, get_player_physics

def get_spacing_area(game, frame_number):
    """
    Calculates convex hull AREA of home and away team for a given frame.
    Useful for analyzing the spacing of teams.

    NOTE: for a 2-D hull, scipy's `ConvexHull.area` is the PERIMETER and
    `ConvexHull.volume` is the enclosed area. Earlier versions of this
    function used `.area`, so any dataset extracted before this fix
    (including data/shot_features_full.csv / _valid2.csv) carries a
    perimeter ratio in `ratio_off_def_hull`, not an area ratio.

    Args:
        game: Game object
        frame_number (int): number of frame in game to calculate team convex hulls

    Returns: tuple of data (home_area, away_area)
    """
    details = game._get_moment_details(frame_number)
    x_pos = np.array(details[1])
    y_pos = np.array(details[2])
    xy_pos = np.column_stack((x_pos, y_pos))
    home_area = ConvexHull(xy_pos[1:6, :]).volume
    away_area = ConvexHull(xy_pos[6:, :]).volume
    return (home_area, away_area)

def get_voronoi_areas(game, frame_number):
    """
    Calculates Voronoi cells for each player and returns the total area
    occupied by the home and away teams, clipped to the court boundaries.
    """
    details = game._get_moment_details(frame_number)
    x_pos = np.array(details[1])
    y_pos = np.array(details[2])
    
    if len(x_pos) < 11:
        return (0.0, 0.0)
        
    player_x = x_pos[1:]
    player_y = y_pos[1:]
    players = np.column_stack((player_x, player_y))
    
    # Mirroring technique to clip Voronoi cells to the rectangle
    mirrors = []
    for p in players:
        mirrors.append([-p[0], p[1]]) # Mirror across x=0
        mirrors.append([2*94 - p[0], p[1]]) # Mirror across x=94
        mirrors.append([p[0], -p[1]]) # Mirror across y=0
        mirrors.append([p[0], -100 - p[1]]) # Mirror across y=-50
        
    points = np.concatenate([players, mirrors])
    vor = Voronoi(points)
    
    home_area = 0.0
    away_area = 0.0
    
    for i in range(10):
        region_idx = vor.point_region[i]
        region_vertices_indices = vor.regions[region_idx]
        
        if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
            region_vertices = vor.vertices[region_vertices_indices]
            cell_area = ConvexHull(region_vertices).volume 
            
            if i < 5: # Home team
                home_area += cell_area
            else: # Away team
                away_area += cell_area
                
    return (home_area, away_area)

def get_space_control(game, frame_number, team='home', resolution=50, use_time=False):
    """
    Calculates space control heatmap using delta-distance or delta-time metric.
    """
    details = game._get_moment_details(frame_number)
    x_pos = np.array(details[1])
    y_pos = np.array(details[2])
    
    if len(x_pos) < 11:
        return None, None, None
        
    home_x = x_pos[1:6]
    home_y = y_pos[1:6]
    away_x = x_pos[6:11]
    away_y = y_pos[6:11]
    
    if use_time:
        physics = get_player_physics(game, frame_number)
        if not physics:
            use_time = False
        else:
            home_vels = [physics[i]['velocity'] for i in range(1, 6)]
            away_vels = [physics[i]['velocity'] for i in range(6, 11)]
    
    x_grid = np.linspace(0, 94, resolution)
    y_grid = np.linspace(-50, 0, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    if team == 'home':
        team_x, team_y = home_x, home_y
        opp_x, opp_y = away_x, away_y
        if use_time:
            team_vels = home_vels
            opp_vels = away_vels
    else:
        team_x, team_y = away_x, away_y
        opp_x, opp_y = home_x, home_y
        if use_time:
            team_vels = away_vels
            opp_vels = home_vels
    
    for i in range(resolution):
        for j in range(resolution):
            target = np.array([X[i, j], Y[i, j]])
            
            if use_time:
                team_times = []
                for k in range(5):
                    pos = np.array([team_x[k], team_y[k]])
                    vel = team_vels[k]
                    t = time_to_reach(pos, vel, target)
                    team_times.append(t)
                
                opp_times = []
                for k in range(5):
                    pos = np.array([opp_x[k], opp_y[k]])
                    vel = opp_vels[k]
                    t = time_to_reach(pos, vel, target)
                    opp_times.append(t)
                
                t_teammate = np.min(team_times)
                t_opponent = np.min(opp_times)
                
                Z[i, j] = t_opponent - t_teammate
            else:
                team_dists = np.sqrt((team_x - target[0])**2 + (team_y - target[1])**2)
                d_teammate = np.min(team_dists)
                
                opp_dists = np.sqrt((opp_x - target[0])**2 + (opp_y - target[1])**2)
                d_opponent = np.min(opp_dists)
                
                Z[i, j] = d_opponent - d_teammate
            
    return X, Y, Z
