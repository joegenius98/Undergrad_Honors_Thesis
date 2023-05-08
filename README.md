# Note: Work in Progress
I will be likely be cleaning up code/scripts/documentation/READMEs in the coming weeks, so stay patient in the meantime. Nevertheless, all work done for this honors thesis is present in this repo.

# kFactorVAE: Self-Supervised Regularization for Better A.I. Disentanglement

This repo. contains all work conducted for my honors thesis project at William & Mary. It contains a few subdirectories for different VAE-based models I investigated, notably kFactorVAE in the `kFactorVAE` folder, Beta-VAE from the `Disentangling` folder, and Beta-TCVAE from the `beta-tcvae` folder.

It borrows from Professor Shao's [ControlVAE](https://github.com/shj1987/ControlVAE-ICML2020) GitHub repository. Note though that I do not use the `ControlVAE` model itself in `kFactorVAE`, although it is an open avenue!

# Reproducing Results
Head over to the [`thesis_dsprites_scripts`](./kFactorVAE/thesis_dsprites_scripts) directory's `README` file.


# Cleaning up Outputs
If the [`outputs`](./kFactorVAE/outputs), [`vis_logs`](./kFactorVAE/vis_logs), and/or the [`checkpoints`](./kFactorVAE/checkpoints) directories ever grow too large, you may run the `group_seeds.sh` script inside one of those directories. 

For the [`graphs`](./kFactorVAE/graphs) directory, run the Python script `combine_seeds.py` instead.


## Useful Things Learned

This is just for personal memory's sake or for your own benefit too, in case you were curious as to the minor techniques I have acquired. 

TODO: update this. 

(1) How to make a forked repository private

- This was initially a challenge since by default, GitHub does not allow you to make a forked repo. private. 
- [Source](https://gist.github.com/0xjac/85097472043b697ab57ba1b1c7530274)

```
# 1. Clone without any remote branches
git clone --bare [folder to repo. fork]

# 2. Create a new private repo. on the GitHub website

# 3. Push repo. contents + all branches 
cd [local repo. fork directory]
git push --mirror [copied link to private. repo you just created] 

# 4. Delete your local repo. fork
rm -rf .git
cd ..
rm -rf [folder to repo. fork]

# 5. Now just clone the private repo., and voila!

```

Why I needed to do this was that I wanted to make a private clone of Professor Shao's ControlVAE
and work with that private clone to not risk my honors thesis work being shown publically.



(2) How the Transpose method works for higher dimensions

- It's sill switching "rows" and "columns." aka $a_{ij} = a_{ji}$
- But each element $a$ of the "matrix" you are transposing might something other than a singular number. An element could now be an entire matrix, an actual row/column, or a tensor. 

Why I needed to learn this was to understand how Prof. Shao implemented the latent variable traversal (`viz_traverse`) and displayed the results to the `visdom` server.


(3) How to Search Within a Highlighted Text Section within Vim

1. Shift + V and select all that you want
2. You can yank or just press `esc` twice
3. `/\%V[your_search_string]` 

(4) Shell scripting

(5) Literature Review
- When bogged down by the technical details, making a slides presentation, where each paper gets only one slide, with abstracts, conclusions, advantages, and disadvantages can maintain engagement with the literature. The literature review and processing a new idea in light of it was by far the most challenging part. 

(6) Logging data for visualizations
- Visdom, CSV filewriting

(7) LaTeX
- Side-by-side plots, images, adjacent text to images to label factors of variation
- `\input` command to separate out what would otherwise be one giant LaTeX file
