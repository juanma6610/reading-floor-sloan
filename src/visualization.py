import os
import sys
import shutil
import subprocess
import warnings
from subprocess import Popen, PIPE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc, Polygon
import seaborn as sns
from scipy.spatial import ConvexHull, Voronoi

from kinematics import get_player_physics
from spatial import get_space_control


# Primary colors for every NBA franchise, keyed by tracking-data abbreviation.
# Picked for good visibility against the tan court used in draw_court().
NBA_TEAM_COLORS = {
    "ATL": "#E03A3E",  # Atlanta Hawks - red
    "BOS": "#007A33",  # Boston Celtics - green
    "BKN": "#000000",  # Brooklyn Nets - black
    "CHA": "#00788C",  # Charlotte Hornets - teal
    "CHI": "#CE1141",  # Chicago Bulls - red
    "CLE": "#860038",  # Cleveland Cavaliers - wine
    "DAL": "#00538C",  # Dallas Mavericks - blue
    "DEN": "#0E2240",  # Denver Nuggets - navy
    "DET": "#1D42BA",  # Detroit Pistons - blue
    "GSW": "#1D428A",  # Golden State Warriors - blue
    "HOU": "#CE1141",  # Houston Rockets - red
    "IND": "#002D62",  # Indiana Pacers - navy
    "LAC": "#C8102E",  # LA Clippers - red
    "LAL": "#552583",  # Los Angeles Lakers - purple
    "MEM": "#5D76A9",  # Memphis Grizzlies - blue
    "MIA": "#98002E",  # Miami Heat - red
    "MIL": "#00471B",  # Milwaukee Bucks - green
    "MIN": "#0C2340",  # Minnesota Timberwolves - navy
    "NOP": "#0C2340",  # New Orleans Pelicans - navy
    "NYK": "#006BB6",  # New York Knicks - blue
    "OKC": "#007AC1",  # Oklahoma City Thunder - blue
    "ORL": "#0077C0",  # Orlando Magic - blue
    "PHI": "#006BB6",  # Philadelphia 76ers - blue
    "PHX": "#1D1160",  # Phoenix Suns - purple
    "POR": "#E03A3E",  # Portland Trail Blazers - red
    "SAC": "#5A2D81",  # Sacramento Kings - purple
    "SAS": "#000000",  # San Antonio Spurs - black (silver/black colorway)
    "TOR": "#CE1141",  # Toronto Raptors - red
    "UTA": "#002B5C",  # Utah Jazz - navy
    "WAS": "#002B5C",  # Washington Wizards - navy
}


def get_team_color(team_abbreviation, fallback="#444444"):
    """Return the primary color for a team abbreviation, or a fallback."""
    if team_abbreviation is None:
        return fallback
    return NBA_TEAM_COLORS.get(str(team_abbreviation).upper(), fallback)


def draw_court(ax=None, color="gray", lw=2, grid=False, zorder=0):
    """
    Helper function to draw court.
    Modified from Savvas Tjortjoglou with contribution from Michael Wheelock.
    """
    if ax is None:
        ax = plt.gca()

    line_color = "black"
    wood_color = "#E8C898"          # Tan/wood for general court floor
    paint_color = "#F4B5B5"         # Light pink for paint and free-throw circle
    inner_paint_color = "#DF4343"   # Dark red for inner key and center circle

    # Court floor (tan/wood)
    outer = Rectangle((0, -50), width=94, height=50, facecolor=wood_color,
                      edgecolor=line_color, zorder=zorder, lw=lw)

    # Outer paint (pink) - 16ft wide key
    l_outer_box = Rectangle((0, -33), 19, 16, lw=lw, facecolor=paint_color,
                            edgecolor=line_color, zorder=zorder+1)
    r_outer_box = Rectangle((75, -33), 19, 16, lw=lw, facecolor=paint_color,
                            edgecolor=line_color, zorder=zorder+1)

    # Free-throw circles (pink fill so the half above the FT line shows)
    l_free_throw = Circle((19, -25), radius=6, lw=lw, facecolor=paint_color,
                          edgecolor=line_color, zorder=zorder+1)
    r_free_throw = Circle((75, -25), radius=6, lw=lw, facecolor=paint_color,
                          edgecolor=line_color, zorder=zorder+1)

    # Inner paint / restricted area (dark red) - 12ft wide
    l_inner_box = Rectangle((0, -31), 19, 12, lw=lw, facecolor=inner_paint_color,
                            edgecolor=line_color, zorder=zorder+2)
    r_inner_box = Rectangle((75, -31), 19, 12, lw=lw, facecolor=inner_paint_color,
                            edgecolor=line_color, zorder=zorder+2)

    # Hoops and backboards
    l_hoop = Circle((5.35, -25), radius=.75, lw=lw, fill=False,
                    color=line_color, zorder=zorder+3)
    r_hoop = Circle((88.65, -25), radius=.75, lw=lw, fill=False,
                    color=line_color, zorder=zorder+3)
    l_backboard = Rectangle((4, -28), 0, 6, lw=lw, color=line_color,
                            zorder=zorder+3)
    r_backboard = Rectangle((90, -28), 0, 6, lw=lw, color=line_color,
                            zorder=zorder+3)

    # 3-point lines
    l_corner_a = Rectangle((0, -3), 14, 0, lw=lw, color=line_color,
                           zorder=zorder+1)
    l_corner_b = Rectangle((0, -47), 14, 0, lw=lw, color=line_color,
                           zorder=zorder+1)
    r_corner_a = Rectangle((80, -3), 14, 0, lw=lw, color=line_color,
                           zorder=zorder+1)
    r_corner_b = Rectangle((80, -47), 14, 0, lw=lw, color=line_color,
                           zorder=zorder+1)
    l_arc = Arc((5, -25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
                color=line_color, zorder=zorder+1)
    r_arc = Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248,
                lw=lw, color=line_color, zorder=zorder+1)

    # Half court line and center circles
    half_court = Rectangle((47, -50), 0, 50, lw=lw, color=line_color,
                           zorder=zorder+1)
    hc_big_circle = Circle((47, -25), radius=6, lw=lw, facecolor=inner_paint_color,
                           edgecolor=line_color, zorder=zorder+1)
    hc_sm_circle = Circle((47, -25), radius=2, lw=lw, facecolor=paint_color,
                          edgecolor=line_color, zorder=zorder+2)

    court_elements = [outer, l_outer_box, r_outer_box,
                      l_free_throw, r_free_throw,
                      l_inner_box, r_inner_box,
                      l_hoop, r_hoop, l_backboard, r_backboard,
                      l_corner_a, l_corner_b, l_arc,
                      r_corner_a, r_corner_b, r_arc,
                      half_court, hc_big_circle, hc_sm_circle]

    for element in court_elements:
        ax.add_patch(element)

    return ax


def draw_roster(ax, game, frame_positions, table_y_top=-51, row_height=3.2,
                col_width=26):
    """
    Draw a roster table below the court showing on-court players grouped
    by team, like the example reference image (SAS / WAS columns).
    """
    home_id = game.home_id
    away_id = game.away_id

    # Build {player_id: (full_name, jersey, team_id)} lookup
    player_info = {}
    for p in game.tracking_data['events'][0]['home']['players']:
        full_name = f"{p['firstname']} {p['lastname']}"
        player_info[p['playerid']] = (full_name, p['jersey'], home_id)
    for p in game.tracking_data['events'][0]['visitor']['players']:
        full_name = f"{p['firstname']} {p['lastname']}"
        player_info[p['playerid']] = (full_name, p['jersey'], away_id)

    # Split currently on-court players by team (skip the ball, team_id == -1)
    home_players, away_players = [], []
    for player in frame_positions:
        team_id, player_id = player[0], player[1]
        if team_id == -1 or player_id not in player_info:
            continue
        name, jersey, _ = player_info[player_id]
        if team_id == home_id:
            home_players.append((name, jersey))
        else:
            away_players.append((name, jersey))

    # Center the two-column table horizontally around half court (x=47)
    total_width = col_width * 2
    away_x = 47 - total_width / 2
    home_x = 47

    away_color = game.team_colors[away_id]
    home_color = game.team_colors[home_id]

    def cell(x, y, width, height, facecolor, alpha=1.0):
        ax.add_patch(Rectangle((x, y), width, height,
                               facecolor=facecolor, edgecolor='black',
                               lw=1, alpha=alpha, zorder=5))

    # Header row (team abbreviation)
    header_y = table_y_top - row_height
    cell(away_x, header_y, col_width, row_height, away_color)
    cell(home_x, header_y, col_width, row_height, home_color)
    ax.text(away_x + col_width / 2, header_y + row_height / 2,
            game.away_team, ha='center', va='center', color='white',
            fontsize=11, fontweight='bold', zorder=6)
    ax.text(home_x + col_width / 2, header_y + row_height / 2,
            game.home_team, ha='center', va='center', color='white',
            fontsize=11, fontweight='bold', zorder=6)

    # Player rows
    for i in range(5):
        row_y = table_y_top - (i + 2) * row_height
        cell(away_x, row_y, col_width, row_height, away_color, alpha=0.85)
        cell(home_x, row_y, col_width, row_height, home_color, alpha=0.85)
        if i < len(away_players):
            name, jersey = away_players[i]
            ax.text(away_x + col_width / 2, row_y + row_height / 2,
                    f"{name} #{jersey}", ha='center', va='center',
                    color='white', fontsize=9, zorder=6)
        if i < len(home_players):
            name, jersey = home_players[i]
            ax.text(home_x + col_width / 2, row_y + row_height / 2,
                    f"{name} #{jersey}", ha='center', va='center',
                    color='white', fontsize=9, zorder=6)

def get_ffmpeg_path():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    conda_ffmpeg = os.path.join(sys.prefix, 'Library', 'bin', 'ffmpeg.exe')
    if os.path.exists(conda_ffmpeg):
        return conda_ffmpeg
    return "ffmpeg"

def plot_frame(game, frame_number, highlight_player=None,
               commentary=True, show_spacing=False, show_spacing_team=None,
               show_velocity=False, show_control=None, use_time_control=False,
               plot_spacing=None, pipe=None):
    """
    Creates an individual frame of game.
    """
    (game_time, x_pos, y_pos, colors, sizes,
     quarter, shot_clock, game_clock, edges,
     universe_time, jerseys) = game._get_moment_details(frame_number,
                                               highlight_player=highlight_player)
    (commentary_script, score) = game._get_commentary(game_time)
    fig = plt.figure(figsize=(12, 8), dpi=80)

    ax = plt.gca()
    draw_court(ax)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    # Fully opaque markers with a white outline so players stand out
    # cleanly against the colored court.
    edge_widths = [max(e, 1.2) for e in edges]
    plt.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=1.0,
                edgecolors='white', linewidths=edge_widths, zorder=4)

    # Add jersey numbers
    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        if jerseys[i]:
            plt.text(x, y, jerseys[i], ha='center', va='center',
                     color='white', fontsize=10, fontweight='bold',
                     zorder=5)

    # Quarter / game clock / shot clock on a single horizontal line above
    # the court so the header stays tight.
    ax.text(47, 2.8,
            f"Q{quarter}     {game_clock}     shot {shot_clock}",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', zorder=6)

    # Draw the roster table below the court
    frame_positions = game.moments.iloc[frame_number].positions
    draw_roster(ax, game, frame_positions)

    plt.xlim(-5, 100)
    plt.ylim(-78, 5)
    sns.set_style('dark')
    if commentary:
        plt.figtext(0.23, -.6, commentary_script, size=20)
    # Score header tight against the top of the court (data coords).
    ax.text(47, 4.4,
            f"{game.away_team}  {score}  {game.home_team}",
            ha='center', va='bottom',
            fontsize=14, fontweight='bold', zorder=6)
    if highlight_player:
        plt.figtext(0.17, .95, highlight_player, size=14)
                   
    if show_spacing == "ch":
        xy_pos = np.column_stack((np.array(x_pos), np.array(y_pos)))
        if show_spacing_team == 'home':
            points = xy_pos[1:6, :]
        if show_spacing_team == 'away':
            points = xy_pos[6:, :]
        hull = ConvexHull(points)
        hull_points = points[hull.vertices, :]
        polygon = Polygon(hull_points, alpha=0.3, color='gray')
        ax.add_patch(polygon)
    
    if show_spacing == 'vor':
        details = game._get_moment_details(frame_number)
        xp = np.array(details[1])
        yp = np.array(details[2])
        
        if len(xp) == 11:
            player_x = xp[1:]
            player_y = yp[1:]
            players = np.column_stack((player_x, player_y))
            
            mirrors = []
            for p in players:
                mirrors.append([-p[0], p[1]])
                mirrors.append([2*94 - p[0], p[1]])
                mirrors.append([p[0], -p[1]])
                mirrors.append([p[0], -100 - p[1]])
            
            points = np.concatenate([players, mirrors])
            vor = Voronoi(points)
            
            for i in range(10):
                region_idx = vor.point_region[i]
                region_vertices_indices = vor.regions[region_idx]
                
                if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
                    region_vertices = vor.vertices[region_vertices_indices]
                    color = 'red' if i < 5 else 'blue'
                    polygon = Polygon(region_vertices, alpha=0.2, facecolor=color, edgecolor='black', lw=1)
                    ax.add_patch(polygon)
    
    if show_velocity and frame_number > 0:
        physics = get_player_physics(game, frame_number)
        
        if physics:
            curr = game._get_moment_details(frame_number)
            xp = np.array(curr[1][1:])
            yp = np.array(curr[2][1:])
            
            dx = np.array([physics[i]['velocity'][0] for i in range(1, 11)])
            dy = np.array([physics[i]['velocity'][1] for i in range(1, 11)])
            speeds = np.array([physics[i]['speed'] for i in range(1, 11)])

            circle_offset = 1.8 
            min_visual_speed = 2.0 
            
            plot_dx = dx.copy()
            plot_dy = dy.copy()
            
            moving_mask = speeds > 0.1
            
            for i in range(len(speeds)):
                if 0.1 < speeds[i] < min_visual_speed:
                    plot_dx[i] = (dx[i] / speeds[i]) * min_visual_speed
                    plot_dy[i] = (dy[i] / speeds[i]) * min_visual_speed

            x_start = xp.copy()
            y_start = yp.copy()
            
            x_start[moving_mask] += (dx[moving_mask] / speeds[moving_mask]) * circle_offset
            y_start[moving_mask] += (dy[moving_mask] / speeds[moving_mask]) * circle_offset

            plt.quiver(x_start, y_start, plot_dx, plot_dy, 
                       angles='xy', scale_units='xy', scale=5, 
                       color='black', width=0.005, headwidth=3, 
                       headlength=5, zorder=5, pivot='tail')
            
            for i, (px, py, speed) in enumerate(zip(xp, yp, speeds)):
                text_x = px + (dx[i]/speeds[i] * 3) if speeds[i] > 0.1 else px + 1
                text_y = py + (dy[i]/speeds[i] * 3) if speeds[i] > 0.1 else py + 1
                
                plt.text(text_x, text_y, f"{speed:.1f} ft/s", 
                         fontsize=8, color='black', alpha=0.8, 
                         fontweight='bold', ha='center')
            
    if show_control:
        control_team = show_control if isinstance(show_control, str) else 'home'
        
        X, Y, Z = get_space_control(game, frame_number, team=control_team, 
                                         resolution=50, use_time=use_time_control)
        
        if X is not None:
            im = ax.imshow(Z, extent=[0, 94, -50, 0], origin='lower',
                          cmap='RdBu_r', alpha=0.5, vmin=-12, vmax=12, zorder=1)
            
    if pipe:
        fig.canvas.draw()
        string = fig.canvas.tostring_argb()
        pipe.stdin.write(string)
        plt.close()
        if commentary:
            fig = plt.figure(figsize=(12, 6), dpi=80)
            plt.figtext(.2, .4, commentary_script, size=20)
            fig.canvas.draw()
            string = fig.canvas.tostring_argb()
            pipe.stdin.write(string)
        plt.close()

    else:
        plt.savefig(f'temp/{frame_number}.png', bbox_inches='tight')
        plt.close()


def animate_play(game, game_time, length, highlight_player=None,
                 commentary=True, show_spacing=None, show_spacing_team=None,
                 show_velocity=False, show_control=None, use_time_control=False):
    """
    Method for animating plays in game.
    """
    if type(game_time) == tuple:
        starting_frame = game_time[0]
        ending_frame = game_time[1]
    else:
        starting_frame = game.moments[game.moments.game_time.round() ==
                                      game_time].index.values[0]
        ending_frame = game.moments[game.moments.game_time.round() ==
                                    game_time + length].index.values[0]

    filename = f"temp/{game_time}.mp4"
    size = (960, 1120) if commentary else (960, 640)
    ffmpeg_cmd = get_ffmpeg_path()
    
    cmdstring = (ffmpeg_cmd,
                 '-y', '-r', '20',
                 '-s', '%dx%d' % size,
                 '-pix_fmt', 'argb',
                 '-f', 'rawvideo',  '-i', '-',
                 '-vcodec', 'libx264', filename)

    print(f"Saving video to {os.path.abspath(filename)}...")
    pipe = Popen(cmdstring, stdin=PIPE)
    for frame in range(starting_frame, ending_frame):
        plot_frame(game, frame, highlight_player=highlight_player,
                        commentary=commentary, show_spacing=show_spacing,
                        show_spacing_team=show_spacing_team,
                        show_velocity=show_velocity, show_control=show_control,
                        use_time_control=use_time_control, pipe=pipe)
    pipe.stdin.close()
    pipe.wait()

def watch_player_actions(game, player_name, action, length=15, max_vids=5):
    """
    Method for viewing all plays a player in the game had of a specified type.
    """
    player_action_times = game._get_player_actions(player_name, action)
    for index, time in enumerate(player_action_times):
        if index == max_vids:
            break
        try:
            print(f"Generating video {index+1}/{len(player_action_times) if max_vids is None else min(max_vids, len(player_action_times))}...")
            animate_play(game, time-length, length,
                            highlight_player=player_name,
                            commentary=True)
        except Exception as e:
            print(f"Error generating video for action at time {time}: {e}")
