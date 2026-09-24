"""
Batch Process Feature Extraction (Multiprocessing)

Processes games in parallel using multiprocessing.
Each worker: downloads game → trims to shots → extracts features → returns DataFrame.
Results are collected and appended to a master CSV.
"""

import os
import gc
import sys
import time
import pandas as pd
from game import Game
from shot_features import load_archetypes, extract_shot_features
import traceback
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _process_single_game(args):
    """
    Worker function for multiprocessing.
    Each worker gets its own temp directory to avoid file collisions.
    
    Args:
        args: tuple of (game_index, game_7z, shooter_map, defender_map, s_base, d_base)
    
    Returns:
        tuple of (game_7z, DataFrame or None, error_message or None)
    """
    game_idx, game_7z, shooter_map, defender_map, s_base, d_base = args
    
    # Each worker gets a unique temp directory
    worker_temp = f'temp/worker_{os.getpid()}'
    os.makedirs(worker_temp, exist_ok=True)
    
    try:
        # Parse game details from filename: 01.13.2016.GSW.at.DEN.7z
        parts = game_7z.split('.')
        date = f"{parts[0]}.{parts[1]}.{parts[2]}"
        away_team = parts[3]
        home_team = parts[5]
        
        # Initialize Game (downloads + extracts) into unique worker temp directory
        game = Game(date, home_team, away_team, game_7z=game_7z, temp_dir=worker_temp, verbose=False)
        
        # Extract features (moments are trimmed inside extract_shot_features)
        df = extract_shot_features(
            game, shooter_map, defender_map,
            s_base, d_base, 
            verbose=False
        )
        
        if df is not None and len(df) > 0:
            #df = one_hot_encode_archetypes(df, n_sc, n_dc)
            return (game_7z, df, None)
        else:
            return (game_7z, None, "No shots")
            
    except Exception as e:
        return (game_7z, None, str(e))
    
    finally:
        # Clean up worker temp files
        try:
            shutil.rmtree(worker_temp)
        except Exception:
            pass
        
        # Force garbage collection to free memory
        gc.collect()


def process_batch(start_idx=0, end_idx=50, output_file='data/shot_features_kaggle.csv',
                  n_workers=None, resume=True):
    """
    Process games in parallel using multiprocessing.

    Writes results INCREMENTALLY (one game at a time) and records each
    processed archive in a `<output>.progress` sidecar, so an interrupted
    run never loses completed work and a rerun resumes where it left off.
    Failed archives are written to `failed_games.log` next to the output.

    Args:
        start_idx: Start index in allgames.txt
        end_idx: End index in allgames.txt
        output_file: Path to output CSV
        n_workers: Number of parallel workers (default: CPU cores - 1)
        resume: If True, skip archives already listed in <output>.progress
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    # Load archetypes once (shared across workers via fork/spawn)
    print("Loading archetypes...")
    s_df, d_df, s_cols, d_cols, s_base, d_base = load_archetypes()
    
    # Read all games
    if not os.path.exists('allgames.txt'):
        print("Error: allgames.txt not found.")
        return
    
    with open('allgames.txt', 'r') as f:
        games = [line.strip() for line in f.readlines() if line.strip()]
    
    total_games = len(games)
    end_idx = min(end_idx, total_games)
    games_slice = games[start_idx:end_idx]
    
    print(f"Processing games {start_idx} to {end_idx} ({len(games_slice)} games)")
    print(f"Using {n_workers} parallel workers")
    print(f"Output: {output_file}")

    shooter_map = s_df.set_index('Player').to_dict('index')
    defender_map = d_df.set_index('Player').to_dict('index')

    # --- Resume support: skip archives already recorded as processed ---
    progress_file = output_file + '.progress'
    out_dir = os.path.dirname(output_file) or '.'
    os.makedirs(out_dir, exist_ok=True)
    failed_log = os.path.join(out_dir, 'failed_games.log')

    done = set()
    if resume and os.path.exists(progress_file):
        with open(progress_file) as pf:
            done = {ln.strip() for ln in pf if ln.strip()}
        if done:
            print(f"Resume: {len(done)} archives already processed "
                  f"(from {progress_file}); skipping them.")

    pending = [(i, g) for i, g in enumerate(games_slice) if g not in done]
    if not pending:
        print("Nothing to do — every game in range is already in the output.")
        return

    # Build task list (only for pending games)
    tasks = [(i, g, shooter_map, defender_map, s_base, d_base) for i, g in pending]

    start_time = time.time()
    success = 0
    failed = 0
    total_shots = 0
    failed_games = []

    # Process with multiprocessing Pool. Write INCREMENTALLY as each game
    # finishes (imap_unordered yields on completion) so an interrupted run
    # never loses completed work; the .progress sidecar lets a rerun resume.
    with Pool(processes=n_workers) as pool:
        iterator = pool.imap_unordered(_process_single_game, tasks, chunksize=1)
        for game_7z, df, error in tqdm(iterator, total=len(tasks),
                                       desc="Processing Games", unit="game",
                                       dynamic_ncols=True):
            if df is not None and len(df) > 0:
                header = not os.path.exists(output_file)
                df.to_csv(output_file, mode='a', index=False, header=header)
                with open(progress_file, 'a') as pf:
                    pf.write(game_7z + '\n')
                success += 1
                total_shots += len(df)
            else:
                failed += 1
                failed_games.append((game_7z, error or 'no shots extracted'))

    elapsed = time.time() - start_time

    # Persist the failure list so you can see exactly which archives didn't
    # make it (genuinely-missing games vs. fixable transient errors).
    if failed_games:
        with open(failed_log, 'w') as f:
            for g, e in failed_games:
                f.write(f"{g}\t{e}\n")

    # Cumulative unique games in the output across any resumed runs.
    cumulative = 0
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            cumulative = len({ln.strip() for ln in pf if ln.strip()})

    print(f"\n{'=' * 60}")
    print(f"Batch processing complete in {elapsed:.1f}s")
    print(f"  Success this run: {success}/{len(tasks)} games")
    print(f"  Failed this run:  {failed}/{len(tasks)} games")
    print(f"  Total shots extracted this run: {total_shots}")
    print(f"  Avg time per game: {elapsed/max(len(tasks),1):.1f}s")
    print(f"  Cumulative games in output: {cumulative}")
    if failed_games:
        print(f"\nFailed games ({len(failed_games)}) -> {failed_log}")
        for g, e in failed_games[:20]:
            print(f"  {g}: {e}")
        if len(failed_games) > 20:
            print(f"  ... and {len(failed_games) - 20} more (see {failed_log})")


def process_batch_sequential(start_idx=0, end_idx=50, output_file='data/shot_features_full.csv'):
    """
    Sequential fallback (original behavior) — useful for debugging.
    """
    print("Loading archetypes...")
    s_df, d_df, s_cols, d_cols, s_base, d_base = load_archetypes()
    shooter_map = s_df.set_index('Player').to_dict('index')
    defender_map = d_df.set_index('Player').to_dict('index')

    if not os.path.exists('allgames.txt'):
        print("Error: allgames.txt not found.")
        return

    with open('allgames.txt', 'r') as f:
        games = [line.strip() for line in f.readlines() if line.strip()]

    total_games = len(games)
    end_idx = min(end_idx, total_games)

    print(f"Processing games {start_idx} to {end_idx} (sequential mode)")
    os.makedirs('temp', exist_ok=True)

    for i in range(start_idx, end_idx):
        game_7z = games[i]
        print(f"\n[{i+1}/{end_idx}] Processing {game_7z}...")
        
        parts = game_7z.split('.')
        date = f"{parts[0]}.{parts[1]}.{parts[2]}"
        away_team = parts[3]
        home_team = parts[5]

        try:
            game = Game(date, home_team, away_team, game_7z=game_7z)
            
            df = extract_shot_features(
                game, shooter_map, defender_map,
                s_base, d_base,
                verbose=True
            )
            
            if df is not None and len(df) > 0:
                header = not os.path.exists(output_file)
                df.to_csv(output_file, mode='a', index=False, header=header)
                print(f"Successfully extracted {len(df)} shots.")
            else:
                print(f"No shots extracted for {game_7z}.")

        except Exception as e:
            print(f"FAILED to process {game_7z}: {e}")
            traceback.print_exc()

        # Cleanup
        for filename in os.listdir('temp'):
            file_path = os.path.join('temp', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'  Failed to delete {file_path}. Reason: {e}')

    print("\nBatch processing complete.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch shot feature extraction')
    parser.add_argument('--start', type=int, default=0, help='Start game index')
    parser.add_argument('--end', type=int, default=10, help='End game index')
    parser.add_argument('--output', type=str, default='data/shot_features_full.csv', 
                        help='Output CSV path')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of parallel workers (default: CPU cores - 1)')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential mode instead of multiprocessing')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Reprocess every game even if it is already in <output>.progress')
    args = parser.parse_args()

    if args.sequential:
        process_batch_sequential(args.start, args.end, args.output)
    else:
        process_batch(args.start, args.end, args.output, args.workers, resume=args.resume)
