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



(2) How the Transpose method works for higher dimensions

- It's sill switching "rows" and "columns." aka $a_{ij} = a_{ji}$
- But each element $a$ of the "matrix" you are transposing might something other than a singular number. An element could now be an entire matrix, an actual row/column, or a tensor. 


