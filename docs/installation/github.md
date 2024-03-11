# Installation via GitHub repository

The most recent version of *ldrestoration* can be installed from the GitHub repository. You can either visit the repository by navigating the link on the top right corner of this webpage and perform a manual installation or follow the steps below:

- **Clone the repository to your local folder:**

Open a terminal and type the following command. You may need to configure your user settings in Git before you clone the repository. For further information, please follow
<a href="https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git" target="_blank">this link</a>.

```bash
git clone https://github.com/abodh/LinDistRestoration.git
```

This will create a local repository with the folder name `ldrestoration` in your current directory.

- **Change the directory to `ldrestoration`:**

```bash
cd ldrestoration
```

- **Install the package:**

```bash
python setup.py install
```
This should install the latest version of `ldrestoration` along with the other third-party python packages on which `ldrestoration` depends. Please install any other packages if you encounter `ImportError` when running the examples.