# kFactorVAE: Self-Supervised Regularization for Better A.I. Disentanglement

This repo. contains all work conducted for my honors thesis project at William & Mary. It contains a few subdirectories for different VAE-based models I investigated adding my regularization term 
into, notably kFactorVAE in the 
[`kFactorVAE`](kFactorVAE) folder, 
Beta-VAE from the [`Disentangling`](Disentangling) folder, and Beta-TCVAE from the 
[`beta-tcvae`](beta-tcvae) folder.


It borrows from Professor Shao's [ControlVAE](https://github.com/shj1987/ControlVAE-ICML2020) GitHub repository. Note though that I do not use the `ControlVAE` model itself in `kFactorVAE`, although it is an open avenue!


## Dependencies
I used a `conda` environment and the `YML` file for its `CUDA` dependencies can be found [here](requirements_CUDA_11.6.yml) and for its non-CUDA dependencies [here](requirements_no_CUDA.yml). 


## Reproducing Thesis Paper Results

### Hardware Setup
If you are affiliated with William & Mary, 
I recommend using the lab machines in McGlothlin-Street (McG) Hall. To remotely log onto
these machines, if you do not possess a 
CS account already and/or you are new to using the machines follow the guidance provided by
Professor Timothy A. Davis [here](https://www.cs.wm.edu/~tadavis/remoteaccess.html#:~:text=To%20request%20a%20CS%20account%2C%20enter%20your%20information%20on%20the,24%20hours%20of%20a%20request.). 

Otherwise, I recommend using a computer that has a NVIDIA GPU
and is compatible with NVIDIA CUDA. The NVIDIA GPU models I used were:

1. NVIDIA RTX A4000 (fastest at around 70 training iterations/second)
2. NVIDIA A40 (middle at around 50-70 training iterations/second)
3. NVIDIA RTX A5000 (slowest at around 30-40 training iterations/second)

I made no typos. Surprisingly, yes, the A4000 is faster than the A5000 even though
the specifications tell me otherwise. 

And of course, all these models are available on the lab machines. GPU specifications for each machine may be found 
[here](https://support.cs.wm.edu/index.php/specs). As years pass by,
these computers will probably have significant upgrades in their GPUs. However, you can
probably expect these GPUs to stick around if you are reading this in the year 2023
or a few years afterwards. But maybe I will be proven wrong. 


### Software Instructions

### [Visdom](https://github.com/fossasia/visdom) Server Initialization

You will need to perform this step if you want to see the training metric graphs, reconstructions, 
and latent traversals when reproducing experimental results on a convenient website interface.
All these results are also stored in directories. 

On a (Linux) shell, run:

```❯ chmod +x run_visdom_server.sh```

```❯ run_visdom_server.sh [port number] [optional relative path to Visdom log file to replay]```

You may replay as many log files as you want.
Keep in mind you will get a `.out` file every time run [`run_visdom_server.sh`](run_visdom_server.sh).

### kFactorVAE Scripts Setup

You may head over to the [kFactorVAE README](kFactorVAE/README.md).

## Technical Support for McG Hall Computers
If you have any issues/requests, please 
reach out to Joseph Hause in the [W&M Computer Science Slack](https://join.slack.com/t/wm-cs/shared_invite/zt-1v4tjn703-1cTnS56msdQBzZwz7VlIqg). 

## Useful Tidbits Learned

This is just for personal memory's sake or for your own benefit too, in case you were curious as to the minor techniques I have acquired. 

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
- conditionals to check directories
- operations on directories
- checking if a port is being used by a specific process
- for loops 
- variables
- realizing that ChatGPT is solid in shell scripting :)

(5) Literature Review
- When bogged down by the technical details, making a slides presentation, where each paper gets only one slide, with abstracts, conclusions, advantages, and disadvantages can maintain engagement with the literature. The literature review and processing a new idea in light of it was by far the most challenging part. 

(6) Logging data for visualizations
- Visdom, CSV filewriting

(7) LaTeX
- Side-by-side plots, images, adjacent text to images to label factors of variation
- `\input` command to separate out what would otherwise be one giant LaTeX file
