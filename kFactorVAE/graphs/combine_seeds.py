"""This file groups together the shared named CSV files across each group defined by, for example,
"...seed1", "...seed2", "...seed3", "...seed4", and "...seed5"

Can accept one user argument: the substring before *seed* in the glob search

assistance from ChatGPT
"""

from pathlib import Path
import sys
import pandas as pd 
import re
import shutil

pattern = re.compile(r".*seed(\d+)$")

base_path = Path(__file__).parent

search_str = "*seed*" if len(sys.argv) == 1 else f"{sys.argv[1]}*seed*"

# get all the directories in the base path
all_dirs = [p for p in base_path.glob(search_str) if p.is_dir() and pattern.match(p.name)]


# group the directories
group_to_dirs = {}
for dir in all_dirs:
    group = dir.name.split("seed")[0]
    if group[-1] == "_":
        group = group[:-1]
    if group not in group_to_dirs:
        group_to_dirs[group] = []
    group_to_dirs[group].append(dir)

# print(group_to_dirs)

# merge files within each group
def merge_df_lst(df_lst):
    if not df_lst:
        return None
    else:
        to_ret = df_lst[0]
        for df in df_lst[1:]:
            to_ret = to_ret.merge(df, on='iteration')
        return to_ret


for group, group_dirs in group_to_dirs.items():
    # ensure that group directories are sorted by seed number in incr. order
    group_dirs.sort(key = lambda dir: int(dir.name.split("seed")[-1]))
    # print(group_dirs)

    # list of pandas dataframes to merge
    # each list contains data across all the seeds in this group
    recon_losses = []
    kl_divs = []
    discrim_accs = []
    k_sim_losses = []
    total_corrs = []

    for group_dir in group_dirs:
        recon_loss_dir = group_dir / 'recon_loss.csv'
        kl_div_path_dir = group_dir / 'kl_divs.csv'
        discrim_acc_dir = group_dir / 'discrim_acc.csv'
        k_sim_loss_dir = group_dir / 'k_sim_loss.csv'
        total_corr_dir = group_dir / 'total_corr.csv'

        if recon_loss_dir.exists():
            recon_losses.append(pd.read_csv(recon_loss_dir))
        else:
            print(f"recon_loss.csv does not exist inside dir. {group}")
        
        if kl_div_path_dir.exists():
            kl_divs.append(pd.read_csv(kl_div_path_dir))
        else:
            print(f"kl_divs.csv does not exist inside dir. {group}")
        
        if discrim_acc_dir.exists():
            discrim_accs.append(pd.read_csv(discrim_acc_dir))
        else:
            print(f"discrim_acc.csv does not exist inside dir. {group}")

        if k_sim_loss_dir.exists():
            k_sim_losses.append(pd.read_csv(k_sim_loss_dir))
        else:
            print(f"k_sim_loss.csv does not exist inside dir. {group}")

        if total_corr_dir.exists():
            total_corrs.append(pd.read_csv(total_corr_dir))
        else:
            print(f"total_corr.csv does not exist inside dir. {group}")
        
        
    # merge the dataframes
    recon_losses_df = merge_df_lst(recon_losses)
    kl_divs_df = merge_df_lst(kl_divs)
    discrim_accs_df = merge_df_lst(discrim_accs)
    k_sim_losses_df = merge_df_lst(k_sim_losses)
    total_corrs_df = merge_df_lst(total_corrs)

    # make the new directory and plug in the merged dataframes inside as CSV files
    new_group_dir = base_path / group
    new_group_dir.mkdir(exist_ok=True)

    recon_losses_df.to_csv(new_group_dir / 'recon_loss.csv', index=False)
    kl_divs_df.to_csv(new_group_dir / 'kl_divs.csv', index=False)
    discrim_accs_df.to_csv(new_group_dir / 'discrim_acc.csv', index=False)
    k_sim_losses_df.to_csv(new_group_dir / 'k_sim_loss.csv', index=False)
    total_corrs_df.to_csv(new_group_dir / 'total_corr.csv', index=False)


# delete the original directories that make up the newly-formed group directories
for group_dirs in group_to_dirs.values():
    for group_dir in group_dirs:
        shutil.rmtree(base_path / group_dir)


        





    
        
