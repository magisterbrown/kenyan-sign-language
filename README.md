
# Best Practices

## Folder Structure
Follow these best practices for the folder structure:
https://dzone.com/articles/data-science-project-folder-structure

- src: Code related to feature engineering, pre-processing, exploration, etc. (may have sub-folders such as: exploration and pre-processing)
	- exploration (exploration will be done in notebooks but you should only commit python scripts. If you think the insights you've gathered are particularly useful, share them through Google Drive.)
	- pre-processing
- tests: Unit tests for code in src
- models: Trained models (each model should have a model_id and a lookup table (a simple csv, Google Sheets, or Excel would work) that specifies all hyper-parameters for that model)
- data: All the data, if it fits
	- train: training and metadata
	- validation: validation data (create it after you have a dedicated validation set, or you can have a sub-folder under train, whichever suits best)
	- test: test data
- pipeline: Scripts to re-train (and test) the model in an automated manner. 
- docs

## Git
Follow these standards:
[Data Science Project Coding Standards](https://datasciencecampus.github.io/coding-standards/version-control.html#git-style-guide)
- Use ```kebab-case``` for branch names
- Specify the type of a commit and explain clearly what the piece of code accomplishes. An example commit:
``` 
feat: Added pre-processing function
<Description of the function>
Closes #12 
```

## Docstrings
You MUST comment each method you've written. All the other team members should at least be able to use your method, if not understand it.
Method comments are strongly encouraged to be written in reStructuredText, the Python standard. To set reStructuredText as your default docstring format in your IDE go to:
- File > Settings > Tools > Python Integrated Tools > Docstrings > Default Docstring Format

Example:
```
:param myParam1: explanation1
:type myParam1: type1
:param myParam2: explanation2
:type myParam2: type2
:return: explanation
```
or simply
```
:param int myParam1: explanation1
:param int myParam2: explanation2
:return str: explanation
```

Specifying the types are optional but recommended.

## Data
Please have the **names of datasets** (.csv, .txt, .whatever) the **same**. Especially if everyone's using the same data for training.
Before preprocessing, **make a copy of the original data and store your preprocessed version under a sub-folder in `/src/preprocessing`**.
If you're certain that the preprocessed version will perform better (or better yet, try it, and if it does perform better, then), consult all your teammates and switch the current version with the preprocessed one.
This might not be very applicable for Kaggle competitions but it's good to keep in mind.

# Computing Power

## DAS-6 Instructions
1. Connect to TU Delft VPN by using your own VPN provider or FortiClient VPN (Recommended). Click 'add a new connection' and put `vpn.tudelft.nl` as the remote gateway. Customize the port as 443. Then log in with your NetID as username and its password.
2. First, connect to the head node by running the command: 
	- `ssh username@fs3.das6.tudelft.nl`
	- Ask the username and password from your team's chief. Don't ever push the password to Git.
3. Now, you can ssh into the machine that we will be working on (Apollo). We will be working on a joint account. Run the following command after you've successfully logged into DAS-6:
	- `ssh epoch2021@10.141.0.101`
	- Ask for the password from your team's chief. Don't ever push the password to Git.

Now you have **128 CPU cores** and **two NVidia 2080Ti** at your disposal!

The disk with the root file system is a fast SSD but has limited capacity. For **larger data**, please place it in **/mnt/hdd** This is a large hard disk drive. 

If you want to use the machine, please send **Jan Rellermeyer - EWI** <[J.S.Rellermeyer@tudelft.nl](mailto:J.S.Rellermeyer@tudelft.nl)> a reservation request so that he can coordinate access since he has 2-3 MSc students who also need it occasionally.

**Bootstrap your environment through virtualenv in your home directory.**
