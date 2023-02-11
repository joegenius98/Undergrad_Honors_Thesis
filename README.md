# Joseph Lee's W&M Undergrad. Honors Thesis Work

Adoption of Prof. Shao's ControlVAE work to investigate better disentanglement methods for more complex datasets compared to `dsprites`


## Useful Things Learned

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
