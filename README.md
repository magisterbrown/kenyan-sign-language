
# Best Practices

## Folder Structure
Follow these best practices for the folder structure:
https://dzone.com/articles/data-science-project-folder-structure

- src: Code related to feature engineering, preprocessing,  etc. (may have sub-folders such as: preprocessing)
	- preprocessing
	- visualisation: Scripts to create exploratory and results oriented visualizations
	- models: Scripts to train models and then use trained models to make predictions
		- predict.py
		- train_model.py
- tests: Unit tests for code in src
- models: Trained models (each model should have a model_id and a lookup table (a simple csv, Google Sheets, or Excel would work) that specifies all hyper-parameters for that model)
- notebooks: Jupyter notebooks. Mainly exploration. Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
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

## Notebooks
Notebook packages like the Jupyter notebook, Beaker notebook, Zeppelin, and other literate programming tools are very effective for exploratory data analysis. However, these tools can be less effective for reproducing an analysis. When we use notebooks in our work, we often subdivide the notebooks folder. For example, `notebooks/exploratory` contains initial explorations, whereas `notebooks/reports` is more polished work that can be exported as html to the reports directory.

Since notebooks are challenging objects for source control (e.g., diffs of the json are often not human-readable and merging is near impossible), we recommended not collaborating directly with others on Jupyter notebooks. There are two steps we recommend for using notebooks effectively:

1. Follow a naming convention that shows the owner and the order the analysis was done in. We use the format <step>-<ghuser>-<description>.ipynb (e.g., 0.3-bull-visualize-distributions.ipynb).
2. Refactor the good parts. Don't write code to do the same task in multiple notebooks. If it's a data preprocessing task, put it in the pipeline at src/data/make_dataset.py and load data from data/interim. If it's useful utility code, refactor it to src.

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
Don't ever edit your raw data, especially not manually, and especially not in Excel. Don't overwrite your raw data. Don't save multiple versions of the raw data. Treat the data (and its format) as immutable. The code you write should move the raw data through a pipeline to your final analysis. You shouldn't have to run all of the steps every time you want to make a new figure (see Analysis is a DAG), but anyone should be able to reproduce the final products with only the code in src and the data in data/raw.

Also, if data is immutable, it doesn't need source control in the same way that code does. Therefore, by default, the data folder is included in the .gitignore file. If you have a small amount of data that rarely changes, you may want to include the data in the repository. Github currently warns if files are over 50MB and rejects files over 100MB. Some other options for storing/syncing large data include AWS S3 with a syncing tool (e.g., s3cmd), Git Large File Storage, Git Annex, and dat. Currently by default, we ask for an S3 bucket and use AWS CLI to sync data in the data folder with the server.


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
